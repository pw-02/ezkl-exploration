import os
import time
from typing import Dict
import ezkl
from pyparsing import Any

def calibrate_settings(onnx_model_path, input_data_path, settings_path):
        print("CALIBRATING")
        # Generate and calibrate settings (no cache hit)
        ezkl.gen_settings(onnx_model_path, settings_path)
        ezkl.calibrate_settings(input_data_path, onnx_model_path, settings_path, "resources")
        assert os.path.exists(settings_path)

def compile_circuit(onnx_model_path, compiled_circuit_path, settings_path):
        print("COMPILING")
 
        # Compile circuit (no cache hit)
        ezkl.compile_circuit(
            onnx_model_path, compiled_circuit_path, settings_path
        )
        assert os.path.exists(compiled_circuit_path)

def get_srs(settings_path):
        print("GETTING_SRS")
        ezkl.get_srs(settings_path)

def gen_witness(input_data_path, compiled_circuit_path, witness_path):
        print("GENERATING_WITNESS")
        ezkl.gen_witness(input_data_path, compiled_circuit_path, witness_path)
        assert os.path.exists(witness_path)

def gen_keys(compiled_circuit_path, vk_path, pk_path):
        # I here, need to generate new keys
        ezkl.setup(compiled_circuit_path, vk_path, pk_path)


def compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path):
        print("PROVING")
        ezkl.prove(witness_path, compiled_circuit_path, pk_path, proof_path, "single")
        assert os.path.exists(proof_path)

def run_proof(onnx_model_path, 
              input_data_path, 
              settings_path,
              compiled_circuit_path,
              witness_path,
              vk_path,
              pk_path,
              proof_path,
              setup_only=False):
        """
        Run all proof pipeline stages in sequence.
        Returns: perf_measurements dict for each stage.
        """

        perf_measurements: Dict[str, Any] = {}
        calibrate_settings_start = time.perf_counter()
        calibrate_settings(onnx_model_path, input_data_path, settings_path)
        calibrate_settings_time = time.perf_counter() - calibrate_settings_start
        perf_measurements["calibrate_settings_time(s)"] = calibrate_settings_time

        compile_circuit_start = time.perf_counter()
        compile_circuit(onnx_model_path, compiled_circuit_path, settings_path)
        compile_circuit_time = time.perf_counter() - compile_circuit_start
        perf_measurements["ezkl_compile_circuit_time(s)"] = compile_circuit_time

        get_srs_start = time.perf_counter()
        get_srs(settings_path)
        get_srs_time = time.perf_counter() - get_srs_start
        perf_measurements["ezkl_get_srs_time(s)"] = get_srs_time
        total_setup_time += get_srs_time

        gen_witness_start = time.perf_counter()
        gen_witness(input_data_path, compiled_circuit_path, witness_path)
        gen_witness_time = time.perf_counter() - gen_witness_start
        perf_measurements["ezkl_gen_witness_time(s)"] = gen_witness_time
        total_setup_time += gen_witness_time

        key_gen_start = time.perf_counter()
        gen_keys(compiled_circuit_path, vk_path, pk_path)
        key_gen_time = time.perf_counter() - key_gen_start
        perf_measurements["ezkl_key_gen_time(s)"] = key_gen_time
        total_setup_time += key_gen_time
        perf_measurements["ezkl_setup_time(s)"] = total_setup_time

        if not setup_only:
            compute_proof_start = time.perf_counter()
            compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path)
            compute_proof_time = time.perf_counter() - compute_proof_start
            perf_measurements["ezkl_proof_time(s)"] = compute_proof_time
        return perf_measurements


if __name__ == "__main__":
        base_path = "ezkl_tmp"

        os.makedirs(base_path, exist_ok=True)

        onnx_model_path = ""
        input_data_path = ""
        settings_path = os.path.join(base_path, "settings.json")
        compiled_circuit_path = os.path.join(base_path, "circuit.json")
        witness_path = os.path.join(base_path, "witness.json")
        vk_path = os.path.join(base_path, "vk.json")
        pk_path = os.path.join(base_path, "pk.json")
        proof_path = os.path.join(base_path, "proof.json")

        metrics=  run_proof(onnx_model_path, 
                  input_data_path, 
                  settings_path,
                  compiled_circuit_path,
                  witness_path,
                  vk_path,
                  pk_path,
                  proof_path)

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
    await run_ezkl(ezkl.gen_settings, onnx_model_path, settings_path)
    await run_ezkl(ezkl.calibrate_settings, input_data_path, onnx_model_path, settings_path, "resources")
    assert os.path.exists(settings_path)


async def compile_circuit(onnx_model_path, compiled_circuit_path, settings_path):
    print("COMPILING")
    await run_ezkl(ezkl.compile_circuit, onnx_model_path, compiled_circuit_path, settings_path)
    assert os.path.exists(compiled_circuit_path)


async def get_srs(settings_path):
    print("GETTING_SRS")
    await run_ezkl(ezkl.get_srs, settings_path)


async def gen_witness(input_data_path, compiled_circuit_path, witness_path):
    print("GENERATING_WITNESS")
    await run_ezkl(ezkl.gen_witness, input_data_path, compiled_circuit_path, witness_path)
    assert os.path.exists(witness_path)


async def gen_keys(compiled_circuit_path, vk_path, pk_path):
    print("GENERATING_KEYS")
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
    base_path = "ezkl_tmp"
    os.makedirs(base_path, exist_ok=True)

    input_data_path = "examples/onnx/mnist_classifier/input.json"
    onnx_model_path = "examples/onnx/mnist_classifier/network.onnx"
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
