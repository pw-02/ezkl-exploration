

import logging
import os
import time
import asyncio
import inspect
from typing import Dict, Any
import ezkl


async def run_ezkl(fn, *args, **kwargs):
    """
    Call an ezkl function safely: if it returns an awaitable, await it;
    otherwise just return the result.
    """
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def calibrate_settings(onnx_model_path, input_data_path, settings_path):
    print("CALIBRATING")
    if not os.path.exists(settings_path):
        await run_ezkl(ezkl.gen_settings, onnx_model_path, settings_path)
        await run_ezkl(ezkl.calibrate_settings, input_data_path, onnx_model_path, settings_path, "resources")
    assert os.path.exists(settings_path)


async def compile_circuit(onnx_model_path, compiled_circuit_path, settings_path):
    print("COMPILING")
    if not os.path.exists(compiled_circuit_path):
        await run_ezkl(ezkl.compile_circuit, onnx_model_path, compiled_circuit_path, settings_path)
    assert os.path.exists(compiled_circuit_path)


async def get_srs(settings_path):
    print("GETTING_SRS")
    await run_ezkl(ezkl.get_srs, settings_path)


async def gen_witness(input_data_path, compiled_circuit_path, witness_path):
    print("GENERATING_WITNESS")
    if not os.path.exists(witness_path):
        await run_ezkl(ezkl.gen_witness, input_data_path, compiled_circuit_path, witness_path)
    assert os.path.exists(witness_path)


async def gen_keys(compiled_circuit_path, vk_path, pk_path):
    print("GENERATING_KEYS")
    if not os.path.exists(vk_path) or not os.path.exists(pk_path):
        await run_ezkl(ezkl.setup, compiled_circuit_path, vk_path, pk_path)


async def compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path):
    print("PROVING")
    await run_ezkl(ezkl.prove, witness_path, compiled_circuit_path, pk_path, proof_path, "single")
    assert os.path.exists(proof_path)


async def run_proof(
    onnx_model_path,
    input_data_path,
    settings_path,
    compiled_circuit_path,
    witness_path,
    vk_path,
    pk_path,
    proof_path,
    setup_only=False,
) -> Dict[str, Any]:
    """
    Run all proof pipeline stages in sequence.
    Returns: perf_measurements dict for each stage.
    """
    perf_measurements: Dict[str, Any] = {}
    total_setup_time = 0.0

    start = time.perf_counter()
    await calibrate_settings(onnx_model_path, input_data_path, settings_path)
    perf_measurements["calibrate_settings_time(s)"] = time.perf_counter() - start

    start = time.perf_counter()
    await compile_circuit(onnx_model_path, compiled_circuit_path, settings_path)
    perf_measurements["ezkl_compile_circuit_time(s)"] = time.perf_counter() - start

    start = time.perf_counter()
    await get_srs(settings_path)
    t = time.perf_counter() - start
    perf_measurements["ezkl_get_srs_time(s)"] = t
    total_setup_time += t

    start = time.perf_counter()
    await gen_witness(input_data_path, compiled_circuit_path, witness_path)
    t = time.perf_counter() - start
    perf_measurements["ezkl_gen_witness_time(s)"] = t
    total_setup_time += t

    start = time.perf_counter()
    await gen_keys(compiled_circuit_path, vk_path, pk_path)
    t = time.perf_counter() - start
    perf_measurements["ezkl_key_gen_time(s)"] = t
    total_setup_time += t
    perf_measurements["ezkl_setup_time(s)"] = total_setup_time

    if not setup_only:
        start = time.perf_counter()
        await compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path)
        perf_measurements["ezkl_proof_time(s)"] = time.perf_counter() - start

    return perf_measurements


if __name__ == "__main__":
    #set debug logger

    logging.basicConfig(level=logging.DEBUG)

    
    base_path = "ezkl_tmp"
    os.makedirs(base_path, exist_ok=True)
    try:
        input_data_path = "examples/onnx/bert/bert_tiny_squad_data.json"
        onnx_model_path = "examples/onnx/bert/bert_tiny_squad.onnx"
        settings_path = os.path.join(base_path, "settings.json")
        compiled_circuit_path = os.path.join(base_path, "circuit.json")
        witness_path = os.path.join(base_path, "witness.json")
        vk_path = os.path.join(base_path, "vk.json")
        pk_path = os.path.join(base_path, "pk.json")
        proof_path = os.path.join(base_path, "proof.json")

        metrics = asyncio.run(
            run_proof(
                onnx_model_path,
                input_data_path,
                settings_path,
                compiled_circuit_path,
                witness_path,
                vk_path,
                pk_path,
                proof_path,
            )
        )

        print("Perf measurements:", metrics)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        #deletebase base_path folder and all its contents
        import shutil
        shutil.rmtree(base_path)