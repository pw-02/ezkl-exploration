#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import ezkl


WORK = Path("ezkl_smoke_test_out")
MODEL = WORK / "model.onnx"
INPUT = WORK / "input.json"
SETTINGS = WORK / "settings.json"
COMPILED = WORK / "network.compiled"
WITNESS = WORK / "witness.json"
PK = WORK / "pk.key"
VK = WORK / "vk.key"
PROOF = WORK / "proof.json"


class TinyModel(nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


def run_step(name, fn):
    print(f"\n=== {name} ===")
    try:
        out = fn()
        print(f"{name}: OK -> {out}")
        return out
    except Exception as e:
        print(f"{name}: FAILED")
        print(repr(e))
        raise


def print_env():
    print("=== ENV ===")
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("ezkl:", getattr(ezkl, "__version__", "unknown"))
    print("torch:", torch.__version__)
    print("HOME:", os.environ.get("HOME"))
    print("USERPROFILE:", os.environ.get("USERPROFILE"))

    cli = shutil.which("ezkl")
    print("ezkl cli:", cli)
    if cli:
        try:
            print(
                "ezkl cli version:",
                subprocess.check_output([cli, "--version"], text=True).strip(),
            )
        except Exception as e:
            print("ezkl cli version failed:", repr(e))


def get_expected_srs_path(settings_path: Path) -> Path:
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    run_args = settings.get("run_args", {})
    logrows = run_args.get("logrows")
    commitment = str(run_args.get("commitment", "KZG")).lower()

    if logrows is None:
        raise ValueError("settings.json missing run_args.logrows")

    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or "."
    srs_dir = Path(home) / ".ezkl" / "srs"
    srs_dir.mkdir(parents=True, exist_ok=True)

    if commitment == "ipa":
        srs_name = f"ipa{logrows}.srs"
    else:
        srs_name = f"kzg{logrows}.srs"

    return srs_dir / srs_name


def print_file_debug(path: Path, label: str):
    print(f"\n=== {label} ===")
    print("path:", path)

    if not path.exists():
        print("exists: false")
        return

    print("exists: true")
    print("size:", path.stat().st_size, "bytes")

    with open(path, "rb") as f:
        header = f.read(64)

    print("first 64 bytes hex:", header.hex())

    try:
        print("file command:", subprocess.check_output(["file", str(path)], text=True).strip())
    except Exception as e:
        print("file command failed:", repr(e))


def main():
    WORK.mkdir(exist_ok=True)
    print_env()

    model = TinyModel().eval()
    x = torch.tensor([[0.25, -1.0, 2.0]], dtype=torch.float32)

    run_step(
        "export_onnx",
        lambda: torch.onnx.export(
            model,
            x,
            MODEL,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes=None,
        ),
    )

    with open(INPUT, "w", encoding="utf-8") as f:
        json.dump({"input_data": [x.detach().numpy().reshape(-1).tolist()]}, f)

    print("model:", MODEL.resolve())
    print("input:", INPUT.resolve())

    run_step(
        "gen_settings",
        lambda: ezkl.gen_settings(
            model=str(MODEL),
            output=str(SETTINGS),
        ),
    )

    run_step(
        "calibrate_settings",
        lambda: ezkl.calibrate_settings(
            data=str(INPUT),
            model=str(MODEL),
            settings=str(SETTINGS),
            target="resources",
        ),
    )

    with open(SETTINGS, "r", encoding="utf-8") as f:
        settings = json.load(f)

    print("\n=== SETTINGS RUN_ARGS ===")
    print(json.dumps(settings.get("run_args", settings), indent=2)[:4000])

    srs_path = get_expected_srs_path(SETTINGS)
    print("\nExpected ezkl SRS path:", srs_path)
    print_file_debug(srs_path, "SRS BEFORE get_srs")

    run_step(
        "compile_circuit",
        lambda: ezkl.compile_circuit(
            model=str(MODEL),
            compiled_circuit=str(COMPILED),
            settings_path=str(SETTINGS),
        ),
    )

    run_step(
        "get_srs_explicit_default_path",
        lambda: ezkl.get_srs(
            settings_path=str(SETTINGS),
            srs_path=str(srs_path),
        ),
    )

    print_file_debug(srs_path, "SRS AFTER get_srs")

    run_step(
        "gen_witness",
        lambda: ezkl.gen_witness(
            data=str(INPUT),
            model=str(COMPILED),
            output=str(WITNESS),
        ),
    )

    run_step(
        "mock",
        lambda: ezkl.mock(
            witness=str(WITNESS),
            model=str(COMPILED),
        ),
    )

    run_step(
        "setup",
        lambda: ezkl.setup(
            model=str(COMPILED),
            vk_path=str(VK),
            pk_path=str(PK),
            srs_path=str(srs_path),
        ),
    )

    run_step(
        "prove",
        lambda: ezkl.prove(
            witness=str(WITNESS),
            model=str(COMPILED),
            pk_path=str(PK),
            proof_path=str(PROOF),
            proof_type="single",
            srs_path=str(srs_path),
        ),
    )

    run_step(
        "verify",
        lambda: ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VK),
            srs_path=str(srs_path),
        ),
    )

    print("\nSUCCESS: ezkl smoke test completed.")
    print(f"Artifacts in: {WORK.resolve()}")


if __name__ == "__main__":
    main()