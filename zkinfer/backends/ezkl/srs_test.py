from pathlib import Path
import os
import shutil
import ezkl

MODEL_PATH = "network.onnx"
COMPILED_MODEL_PATH = "network.compiled"
SETTINGS_PATH = "settings.json"
CALIBRATION_DATA = "calibration.json"

EZKL_DIR = Path.home() / ".ezkl"
SRS_DIR = EZKL_DIR / "srs"
SRS_PATH = SRS_DIR / "kzg.srs"

VK_PATH = "vk.key"
PK_PATH = "pk.key"


def file_info(path):
    p = Path(path)
    if p.exists():
        print(f"{p}: exists, size={p.stat().st_size} bytes")
    else:
        print(f"{p}: missing")


def main():
    print("ezkl version:", getattr(ezkl, "__version__", "unknown"))

    SRS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== paths ===")
    print("home:", Path.home())
    print("srs_path:", SRS_PATH)

    print("\n=== clean old SRS ===")
    if SRS_PATH.exists():
        print("Removing old SRS:", SRS_PATH)
        SRS_PATH.unlink()

    print("\n=== gen settings ===")
    res = ezkl.gen_settings(
        model=MODEL_PATH,
        output=SETTINGS_PATH,
    )
    print("gen_settings:", res)
    file_info(SETTINGS_PATH)

    print("\n=== calibrate settings ===")
    res = ezkl.calibrate_settings(
        data=CALIBRATION_DATA,
        model=MODEL_PATH,
        settings=SETTINGS_PATH,
        target="resources",
    )
    print("calibrate_settings:", res)
    file_info(SETTINGS_PATH)

    print("\n=== compile circuit ===")
    res = ezkl.compile_circuit(
        model=MODEL_PATH,
        compiled_model=COMPILED_MODEL_PATH,
        settings_path=SETTINGS_PATH,
    )
    print("compile_circuit:", res)
    file_info(COMPILED_MODEL_PATH)

    print("\n=== get srs ===")
    res = ezkl.get_srs(
        settings_path=SETTINGS_PATH,
        srs_path=str(SRS_PATH),
    )
    print("get_srs:", res)
    file_info(SRS_PATH)

    print("\n=== setup ===")
    res = ezkl.setup(
        model=COMPILED_MODEL_PATH,
        vk_path=VK_PATH,
        pk_path=PK_PATH,
        srs_path=str(SRS_PATH),
    )
    print("setup:", res)
    file_info(VK_PATH)
    file_info(PK_PATH)


if __name__ == "__main__":
    main()