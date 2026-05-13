import asyncio
import inspect
import json
import logging
import os
import shutil
import time
from typing import Any, Callable, Dict

import ezkl


REQUIRED_RUN_ARGS = {"decomp_base", "decomp_legs"}


def setup_logger(log_file: str, level: int = logging.DEBUG) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger("ezkl_debug")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


async def ezkl_call(fn: Callable, *args, **kwargs):
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        res = await res
    return res


def delete_if_exists(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def settings_needs_regen(settings_path: str) -> bool:
    if not os.path.exists(settings_path):
        return True

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        run_args = settings.get("run_args", {})
        return not REQUIRED_RUN_ARGS.issubset(run_args.keys())

    except Exception:
        return True


def clean_dependent_artifacts(
    compiled_circuit_path: str,
    witness_path: str,
    vk_path: str,
    pk_path: str,
    proof_path: str,
) -> None:
    for path in [
        compiled_circuit_path,
        witness_path,
        vk_path,
        pk_path,
        proof_path,
    ]:
        delete_if_exists(path)


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


async def calibrate_settings(
    onnx_model_path: str,
    input_data_path: str,
    settings_path: str,
    logger: logging.Logger,
    run_calibrate: bool = False,
) -> bool:
    logger.info("CALIBRATING")
    logger.info("ONNX model: %s", onnx_model_path)
    logger.info("Input data: %s", input_data_path)
    logger.info("Settings path: %s", settings_path)

    regenerated = False

    if settings_needs_regen(settings_path):
        logger.warning("Regenerating settings for current ezkl version")

        delete_if_exists(settings_path)

        res = await ezkl_call(
            ezkl.gen_settings,
            model=onnx_model_path,
            output=settings_path,
        )
        logger.info("gen_settings result: %s", res)

        regenerated = True

        if run_calibrate:
            res = await ezkl_call(
                ezkl.calibrate_settings,
                data=input_data_path,
                model=onnx_model_path,
                settings=settings_path,
                target="resources",
            )
            logger.info("calibrate_settings result: %s", res)
    else:
        logger.info("Using existing settings: %s", settings_path)

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file not created: {settings_path}")

    dump_json_if_exists(settings_path, logger, "settings.json")
    return regenerated


async def compile_circuit(
    onnx_model_path: str,
    compiled_circuit_path: str,
    settings_path: str,
    logger: logging.Logger,
    force: bool = False,
) -> None:
    logger.info("COMPILING")

    if force:
        delete_if_exists(compiled_circuit_path)

    if not os.path.exists(compiled_circuit_path):
        res = await ezkl_call(
            ezkl.compile_circuit,
            model=onnx_model_path,
            compiled_circuit=compiled_circuit_path,
            settings_path=settings_path,
        )
        logger.info("compile_circuit result: %s", res)
    else:
        logger.info("Using existing compiled circuit: %s", compiled_circuit_path)

    if not os.path.exists(compiled_circuit_path):
        raise FileNotFoundError(f"Compiled circuit not created: {compiled_circuit_path}")


async def get_srs(settings_path: str, logger: logging.Logger) -> None:
    logger.info("GETTING_SRS")
    dump_json_if_exists(settings_path, logger, "settings before get_srs")

    res = await ezkl_call(
        ezkl.get_srs,
        settings_path=settings_path,
    )
    logger.info("get_srs result: %s", res)


async def gen_witness(
    input_data_path: str,
    compiled_circuit_path: str,
    witness_path: str,
    logger: logging.Logger,
    force: bool = False,
) -> None:
    logger.info("GENERATING_WITNESS")

    if force:
        delete_if_exists(witness_path)

    if not os.path.exists(witness_path):
        res = await ezkl_call(
            ezkl.gen_witness,
            data=input_data_path,
            model=compiled_circuit_path,
            output=witness_path,
        )
        logger.info("gen_witness result: %s", res)
    else:
        logger.info("Using existing witness: %s", witness_path)

    if not os.path.exists(witness_path):
        raise FileNotFoundError(f"Witness not created: {witness_path}")


async def gen_keys(
    compiled_circuit_path: str,
    vk_path: str,
    pk_path: str,
    logger: logging.Logger,
    force: bool = False,
) -> None:
    logger.info("GENERATING_KEYS")

    if force:
        delete_if_exists(vk_path)
        delete_if_exists(pk_path)

    if not os.path.exists(vk_path) or not os.path.exists(pk_path):
        res = await ezkl_call(
            ezkl.setup,
            model=compiled_circuit_path,
            vk_path=vk_path,
            pk_path=pk_path,
        )
        logger.info("setup result: %s", res)
    else:
        logger.info("Using existing keys: %s / %s", vk_path, pk_path)

    if not os.path.exists(vk_path):
        raise FileNotFoundError(f"VK not created: {vk_path}")
    if not os.path.exists(pk_path):
        raise FileNotFoundError(f"PK not created: {pk_path}")


async def compute_proof(
    witness_path: str,
    compiled_circuit_path: str,
    pk_path: str,
    proof_path: str,
    logger: logging.Logger,
    force: bool = False,
) -> None:
    logger.info("PROVING")

    if force:
        delete_if_exists(proof_path)

    res = await ezkl_call(
        ezkl.prove,
        witness=witness_path,
        model=compiled_circuit_path,
        pk_path=pk_path,
        proof_path=proof_path,
    )
    logger.info("prove result: %s", res)

    if not os.path.exists(proof_path):
        raise FileNotFoundError(f"Proof not created: {proof_path}")


async def timed_stage(
    name: str,
    logger: logging.Logger,
    fn: Callable,
    *args,
    **kwargs,
) -> float:
    logger.info("Starting stage: %s", name)
    start = time.perf_counter()

    try:
        await fn(*args, **kwargs)
    except Exception:
        logger.exception("Stage failed: %s", name)
        raise

    elapsed = time.perf_counter() - start
    logger.info("Finished stage: %s in %.3fs", name, elapsed)
    return elapsed


async def run_proof(
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
    run_calibrate: bool = False,
    clean: bool = True,
) -> Dict[str, Any]:
    perf_measurements: Dict[str, Any] = {}
    total_setup_time = 0.0

    if clean:
        logger.warning("Cleaning all ezkl artifacts")
        for path in [
            settings_path,
            compiled_circuit_path,
            witness_path,
            vk_path,
            pk_path,
            proof_path,
        ]:
            delete_if_exists(path)

    settings_regenerated_holder = {"value": False}

    async def calibrate_wrapper():
        settings_regenerated_holder["value"] = await calibrate_settings(
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            settings_path=settings_path,
            logger=logger,
            run_calibrate=run_calibrate,
        )

    perf_measurements["calibrate_settings_time(s)"] = await timed_stage(
        "calibrate_settings",
        logger,
        calibrate_wrapper,
    )

    if settings_regenerated_holder["value"]:
        logger.warning("Settings changed; deleting dependent ezkl artifacts")
        clean_dependent_artifacts(
            compiled_circuit_path,
            witness_path,
            vk_path,
            pk_path,
            proof_path,
        )

    perf_measurements["ezkl_compile_circuit_time(s)"] = await timed_stage(
        "compile_circuit",
        logger,
        compile_circuit,
        onnx_model_path,
        compiled_circuit_path,
        settings_path,
        logger,
        force=False,
    )

    t = await timed_stage("get_srs", logger, get_srs, settings_path, logger)
    perf_measurements["ezkl_get_srs_time(s)"] = t
    total_setup_time += t

    t = await timed_stage(
        "gen_witness",
        logger,
        gen_witness,
        input_data_path,
        compiled_circuit_path,
        witness_path,
        logger,
        force=False,
    )
    perf_measurements["ezkl_gen_witness_time(s)"] = t
    total_setup_time += t

    t = await timed_stage(
        "gen_keys",
        logger,
        gen_keys,
        compiled_circuit_path,
        vk_path,
        pk_path,
        logger,
        force=False,
    )
    perf_measurements["ezkl_key_gen_time(s)"] = t
    total_setup_time += t

    perf_measurements["ezkl_setup_time(s)"] = total_setup_time

    if not setup_only:
        perf_measurements["ezkl_proof_time(s)"] = await timed_stage(
            "compute_proof",
            logger,
            compute_proof,
            witness_path,
            compiled_circuit_path,
            pk_path,
            proof_path,
            logger,
            force=True,
        )

    return perf_measurements


async def main() -> None:
    base_path = "_ezkl_tmp"
    os.makedirs(base_path, exist_ok=True)

    logger = setup_logger(os.path.join(base_path, "debug.log"))

    onnx_file = "experiments/models/nanoGPT/network.onnx"
    input_file = "experiments/models/nanoGPT/input.json"


    settings_path = os.path.join(base_path, "settings.json")
    compiled_circuit_path = os.path.join(base_path, "circuit.json")
    witness_path = os.path.join(base_path, "witness.json")
    vk_path = os.path.join(base_path, "vk.json")
    pk_path = os.path.join(base_path, "pk.json")
    proof_path = os.path.join(base_path, "proof.json")

    metrics = await run_proof(
        onnx_model_path=onnx_file,
        input_data_path=input_file,
        settings_path=settings_path,
        compiled_circuit_path=compiled_circuit_path,
        witness_path=witness_path,
        vk_path=vk_path,
        pk_path=pk_path,
        proof_path=proof_path,
        logger=logger,
        setup_only=False,
        run_calibrate=False,
        clean=True,
    )

    logger.info("Perf measurements: %s", metrics)


if __name__ == "__main__":
    asyncio.run(main())