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

    cli = shutil.which("ezkl")
    print("ezkl cli:", cli)
    if cli:
        try:
            print("ezkl cli version:", subprocess.check_output([cli, "--version"], text=True).strip())
        except Exception as e:
            print("ezkl cli version failed:", repr(e))


def main():
    WORK.mkdir(exist_ok=True)
    print_env()

    # 1. Export tiny ONNX model
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

    # 2. ezkl input format
    with open(INPUT, "w") as f:
        json.dump({"input_data": [x.detach().numpy().reshape(-1).tolist()]}, f)

    print("model:", MODEL.resolve())
    print("input:", INPUT.resolve())

    # 3. Generate settings
    run_step(
        "gen_settings",
        lambda: ezkl.gen_settings(
            model=str(MODEL),
            output=str(SETTINGS),
        ),
    )

    # 4. Calibrate settings
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

    print("\n=== SETTINGS RUN_ARGS ===")
    print(json.dumps(settings.get("run_args", settings), indent=2)[:4000])

    # 5. Compile circuit
    run_step(
        "compile_circuit",
        lambda: ezkl.compile_circuit(
            model=str(MODEL),
            compiled_circuit=str(COMPILED),
            settings_path=str(SETTINGS),
        ),
    )

    # 6. SRS
    # This is the step failing on your other machine.
    run_step(
        "get_srs",
        lambda: ezkl.get_srs(
            settings_path=str(SETTINGS),
        ),
    )

    # 7. Witness
    run_step(
        "gen_witness",
        lambda: ezkl.gen_witness(
            data=str(INPUT),
            model=str(COMPILED),
            output=str(WITNESS),
        ),
    )

    # 8. Mock test, catches many circuit/model issues without full proving
    run_step(
        "mock",
        lambda: ezkl.mock(
            witness=str(WITNESS),
            model=str(COMPILED),
        ),
    )

    # 9. Setup keys
    run_step(
        "setup",
        lambda: ezkl.setup(
            model=str(COMPILED),
            vk_path=str(VK),
            pk_path=str(PK),
        ),
    )

    # 10. Prove
    run_step(
        "prove",
        lambda: ezkl.prove(
            witness=str(WITNESS),
            model=str(COMPILED),
            pk_path=str(PK),
            proof_path=str(PROOF),
            proof_type="single",
        ),
    )

    # 11. Verify
    run_step(
        "verify",
        lambda: ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VK),
        ),
    )

    print("\nSUCCESS: ezkl smoke test completed.")
    print(f"Artifacts in: {WORK.resolve()}")


if __name__ == "__main__":
    main()