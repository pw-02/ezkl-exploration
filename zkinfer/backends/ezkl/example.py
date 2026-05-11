import json
import logging
import os
import time
from typing import Any, Dict, Optional

import ezkl


def setup_logger(log_file: str, level: int = logging.DEBUG) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger("ezkl_debug")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def dump_json_if_exists(path: str, logger: logging.Logger, label: str) -> None:
    if not os.path.exists(path):
        logger.warning("%s not found: %s", label, path)
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        logger.debug("%s at %s:\n%s", label, path, json.dumps(data, indent=2)[:8000])

    except Exception as exc:
        logger.warning("Could not dump %s at %s: %s", label, path, exc)


def calibrate_settings(
    onnx_model_path: str,
    input_data_path: str,
    settings_path: str,
    logger: logging.Logger,
    run_calibrate: bool = False,
) -> None:
    logger.info("CALIBRATING")
    logger.info("ONNX model: %s", onnx_model_path)
    logger.info("Input data: %s", input_data_path)
    logger.info("Settings path: %s", settings_path)

    if not os.path.exists(settings_path):
        res = ezkl.gen_settings(onnx_model_path, settings_path)
        logger.info("gen_settings result: %s", res)
        if run_calibrate:
            res = ezkl.calibrate_settings(
                input_data_path,
                onnx_model_path,
                settings_path,
                "resources",
            )
            logger.info("calibrate_settings result: %s", res)
    else:
        logger.info("Using existing settings: %s", settings_path)

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file not created: {settings_path}")

    dump_json_if_exists(settings_path, logger, "settings.json")


def compile_circuit(
    onnx_model_path: str,
    compiled_circuit_path: str,
    settings_path: str,
    logger: logging.Logger,
) -> None:
    logger.info("COMPILING")

    if not os.path.exists(compiled_circuit_path):
        res = ezkl.compile_circuit(
            onnx_model_path,
            compiled_circuit_path,
            settings_path,
        )
        logger.info("compile_circuit result: %s", res)
    else:
        logger.info("Using existing compiled circuit: %s", compiled_circuit_path)

    if not os.path.exists(compiled_circuit_path):
        raise FileNotFoundError(f"Compiled circuit not created: {compiled_circuit_path}")


def get_srs(settings_path: str, logger: logging.Logger) -> None:
    logger.info("GETTING_SRS")
    dump_json_if_exists(settings_path, logger, "settings before get_srs")

    res = ezkl.get_srs(settings_path)
    logger.info("get_srs result: %s", res)


def gen_witness(
    input_data_path: str,
    compiled_circuit_path: str,
    witness_path: str,
    logger: logging.Logger,
) -> None:
    logger.info("GENERATING_WITNESS")

    if not os.path.exists(witness_path):
        res = ezkl.gen_witness(
            input_data_path,
            compiled_circuit_path,
            witness_path,
        )
        logger.info("gen_witness result: %s", res)
    else:
        logger.info("Using existing witness: %s", witness_path)

    if not os.path.exists(witness_path):
        raise FileNotFoundError(f"Witness not created: {witness_path}")


def gen_keys(
    compiled_circuit_path: str,
    vk_path: str,
    pk_path: str,
    logger: logging.Logger,
) -> None:
    logger.info("GENERATING_KEYS")

    if not os.path.exists(vk_path) or not os.path.exists(pk_path):
        res = ezkl.setup(compiled_circuit_path, vk_path, pk_path)
        logger.info("setup result: %s", res)
    else:
        logger.info("Using existing keys: %s / %s", vk_path, pk_path)


def compute_proof(
    witness_path: str,
    compiled_circuit_path: str,
    pk_path: str,
    proof_path: str,
    logger: logging.Logger,
) -> None:
    logger.info("PROVING")

    res = ezkl.prove(
        witness_path,
        compiled_circuit_path,
        pk_path,
        proof_path,
        "single",
    )
    logger.info("prove result: %s", res)

    if not os.path.exists(proof_path):
        raise FileNotFoundError(f"Proof not created: {proof_path}")


def timed_stage(
    name: str,
    logger: logging.Logger,
    fn,
    *args,
    **kwargs,
) -> float:
    logger.info("Starting stage: %s", name)
    start = time.perf_counter()

    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception("Stage failed: %s", name)
        raise

    elapsed = time.perf_counter() - start
    logger.info("Finished stage: %s in %.3fs", name, elapsed)
    return elapsed


def run_proof(
    onnx_model_path: str,
    input_data_path: str,
    settings_path: str,
    compiled_circuit_path: str,
    witness_path: str,
    vk_path: str,
    pk_path: str,
    proof_path: str,
    logger: logging.Logger,
    setup_only: bool = False,
) -> Dict[str, Any]:
    perf_measurements: Dict[str, Any] = {}
    total_setup_time = 0.0

    perf_measurements["calibrate_settings_time(s)"] = timed_stage(
        "calibrate_settings",
        logger,
        calibrate_settings,
        onnx_model_path,
        input_data_path,
        settings_path,
        logger,
    )

    perf_measurements["ezkl_compile_circuit_time(s)"] = timed_stage(
        "compile_circuit",
        logger,
        compile_circuit,
        onnx_model_path,
        compiled_circuit_path,
        settings_path,
        logger,
    )

    t = timed_stage("get_srs", logger, get_srs, settings_path, logger)
    perf_measurements["ezkl_get_srs_time(s)"] = t
    total_setup_time += t

    t = timed_stage(
        "gen_witness",
        logger,
        gen_witness,
        input_data_path,
        compiled_circuit_path,
        witness_path,
        logger,
    )
    perf_measurements["ezkl_gen_witness_time(s)"] = t
    total_setup_time += t

    t = timed_stage(
        "gen_keys",
        logger,
        gen_keys,
        compiled_circuit_path,
        vk_path,
        pk_path,
        logger,
    )
    perf_measurements["ezkl_key_gen_time(s)"] = t
    total_setup_time += t

    perf_measurements["ezkl_setup_time(s)"] = total_setup_time

    if not setup_only:
        perf_measurements["ezkl_proof_time(s)"] = timed_stage(
            "compute_proof",
            logger,
            compute_proof,
            witness_path,
            compiled_circuit_path,
            pk_path,
            proof_path,
            logger,
        )

    return perf_measurements


if __name__ == "__main__":
    base_path = "_ezkl_tmp"
    os.makedirs(base_path, exist_ok=True)

    logger = setup_logger(os.path.join(base_path, "debug.log"))

    try:
        input_data_path = "cache/debug_split/input_1.json"
        onnx_model_path = "cache/debug_split/model_1.onnx"

        settings_path = os.path.join(base_path, "settings.json")
        compiled_circuit_path = os.path.join(base_path, "circuit.json")
        witness_path = os.path.join(base_path, "witness.json")
        vk_path = os.path.join(base_path, "vk.json")
        pk_path = os.path.join(base_path, "pk.json")
        proof_path = os.path.join(base_path, "proof.json")

        metrics = run_proof(
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            settings_path=settings_path,
            compiled_circuit_path=compiled_circuit_path,
            witness_path=witness_path,
            vk_path=vk_path,
            pk_path=pk_path,
            proof_path=proof_path,
            logger=logger,
        )

        logger.info("Perf measurements: %s", metrics)

    except Exception:
        logger.exception("Proof debug run failed")