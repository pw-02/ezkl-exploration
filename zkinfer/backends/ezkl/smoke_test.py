#!/usr/bin/env python3
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import ezkl


WORK = Path("ezkl_smoke_test_out").resolve()
MODEL = WORK / "model.onnx"
INPUT = WORK / "input.json"
SETTINGS = WORK / "settings.json"
COMPILED = WORK / "network.compiled"
WITNESS = WORK / "witness.json"
PK = WORK / "pk.key"
VK = WORK / "vk.key"
PROOF = WORK / "proof.json"
SRS = WORK / "kzg.srs"


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


def file_info(path):
    path = Path(path)
    if path.exists():
        print(f"{path.name}: exists, size={path.stat().st_size} bytes")
    else:
        print(f"{path.name}: MISSING")


def print_env():
    print("=== ENV ===")
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("ezkl python:", getattr(ezkl, "__version__", "unknown"))
    print("torch:", torch.__version__)

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


def call_compile():
    # Newer ezkl Python docs use compiled_circuit.
    try:
        return ezkl.compile_circuit(
            model=str(MODEL),
            compiled_circuit=str(COMPILED),
            settings_path=str(SETTINGS),
        )
    except TypeError:
        # Some older versions used compiled_model.
        return ezkl.compile_circuit(
            model=str(MODEL),
            compiled_model=str(COMPILED),
            settings_path=str(SETTINGS),
        )


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    print_env()
    print("\n=== PATHS ===")
    print("work:", WORK)
    print("srs:", SRS)

    model = TinyModel().eval()
    x = torch.tensor([[0.25, -1.0, 2.0]], dtype=torch.float32)

    run_step(
        "export_onnx",
        lambda: torch.onnx.export(
            model,
            x,
            str(MODEL),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        ),
    )
    file_info(MODEL)

    # EZKL input format: flattened input tensor.
    INPUT.write_text(json.dumps({"input_data": [x.detach().numpy().reshape(-1).tolist()]}))
    file_info(INPUT)

    run_step(
        "gen_settings",
        lambda: ezkl.gen_settings(
            model=str(MODEL),
            output=str(SETTINGS),
        ),
    )
    file_info(SETTINGS)

    run_step(
        "calibrate_settings",
        lambda: ezkl.calibrate_settings(
            data=str(INPUT),
            model=str(MODEL),
            settings=str(SETTINGS),
            target="resources",
        ),
    )
    file_info(SETTINGS)

    print("\n=== SETTINGS PREVIEW ===")
    settings = json.loads(SETTINGS.read_text())
    print(json.dumps(settings.get("run_args", settings), indent=2)[:3000])

    run_step("compile_circuit", call_compile)
    file_info(COMPILED)

    # Critical part: use a fresh explicit SRS file, not ~/.ezkl cache.
    if SRS.exists():
        SRS.unlink()

    run_step(
        "get_srs",
        lambda: ezkl.get_srs(
            settings_path=str(SETTINGS),
            srs_path=str(SRS),
        ),
    )
    file_info(SRS)

    run_step(
        "gen_witness",
        lambda: ezkl.gen_witness(
            data=str(INPUT),
            model=str(COMPILED),
            output=str(WITNESS),
        ),
    )
    file_info(WITNESS)

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
            srs_path=str(SRS),
        ),
    )
    file_info(VK)
    file_info(PK)

    run_step(
        "prove",
        lambda: ezkl.prove(
            witness=str(WITNESS),
            model=str(COMPILED),
            pk_path=str(PK),
            proof_path=str(PROOF),
            srs_path=str(SRS),
        ),
    )
    file_info(PROOF)

    run_step(
        "verify",
        lambda: ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VK),
            srs_path=str(SRS),
        ),
    )

    print("\nSUCCESS: ezkl smoke test completed.")
    print("Artifacts:", WORK)


if __name__ == "__main__":
    main()