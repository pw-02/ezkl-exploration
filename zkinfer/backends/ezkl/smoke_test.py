#!/usr/bin/env python3
import json
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
SRS = WORK / "local_test.srs"


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


def main():
    WORK.mkdir(exist_ok=True)

    print("=== ENV ===")
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("ezkl python:", getattr(ezkl, "__version__", "unknown"))
    print("torch:", torch.__version__)

    cli = shutil.which("ezkl")
    print("ezkl cli:", cli)
    if cli:
        try:
            print("ezkl cli version:", subprocess.check_output([cli, "--version"], text=True).strip())
        except Exception as e:
            print("ezkl cli version failed:", repr(e))

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
        ),
    )

    with open(INPUT, "w") as f:
        json.dump({"input_data": [x.detach().numpy().reshape(-1).tolist()]}, f)

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

    with open(SETTINGS) as f:
        settings = json.load(f)

    print("\n=== RUN_ARGS ===")
    print(json.dumps(settings.get("run_args", {}), indent=2))

    logrows = settings["run_args"]["logrows"]
    print("Using logrows:", logrows)

    run_step(
        "compile_circuit",
        lambda: ezkl.compile_circuit(
            model=str(MODEL),
            compiled_circuit=str(COMPILED),
            settings_path=str(SETTINGS),
        ),
    )

    # IMPORTANT:
    # This avoids ezkl.get_srs(), which downloads/parses public SRS.
    # gen_srs is local and good for debugging/smoke tests.
    run_step(
        "gen_local_srs",
        lambda: ezkl.gen_srs(
            srs_path=str(SRS),
            logrows=logrows,
        ),
    )

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
            srs_path=str(SRS),
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
            srs_path=str(SRS),
        ),
    )

    run_step(
        "verify",
        lambda: ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VK),
            srs_path=str(SRS),
        ),
    )

    print("\nSUCCESS: ezkl local-SRS smoke test completed.")
    print("Artifacts:", WORK.resolve())


if __name__ == "__main__":
    main()