import logging
import os
import time
import ezkl
from typing import Dict, Any


def calibrate_settings(onnx_model_path, input_data_path, settings_path):
    print("CALIBRATING")
    if not os.path.exists(settings_path):
        ezkl.gen_settings(onnx_model_path, settings_path)
        ezkl.calibrate_settings(input_data_path, onnx_model_path, settings_path, "resources")
    assert os.path.exists(settings_path)


def compile_circuit(onnx_model_path, compiled_circuit_path, settings_path):
    print("COMPILING")
    if not os.path.exists(compiled_circuit_path):
        ezkl.compile_circuit(onnx_model_path, compiled_circuit_path, settings_path)
    assert os.path.exists(compiled_circuit_path)


def get_srs(settings_path):
    print("GETTING_SRS")
    ezkl.get_srs(settings_path)


def gen_witness(input_data_path, compiled_circuit_path, witness_path):
    print("GENERATING_WITNESS")
    if not os.path.exists(witness_path):
        ezkl.gen_witness(input_data_path, compiled_circuit_path, witness_path)
    assert os.path.exists(witness_path)


def gen_keys(compiled_circuit_path, vk_path, pk_path):
    print("GENERATING_KEYS")
    if not os.path.exists(vk_path) or not os.path.exists(pk_path):
        ezkl.setup(compiled_circuit_path, vk_path, pk_path)


def compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path):
    print("PROVING")
    ezkl.prove(witness_path, compiled_circuit_path, pk_path, proof_path, "single")
    assert os.path.exists(proof_path)


def run_proof(
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
    calibrate_settings(onnx_model_path, input_data_path, settings_path)
    t = time.perf_counter() - start
    print("CALIBRATED took", t)
    perf_measurements["calibrate_settings_time(s)"] = t

    start = time.perf_counter()
    compile_circuit(onnx_model_path, compiled_circuit_path, settings_path)
    t = time.perf_counter() - start
    print("COMPILED took", t)
    perf_measurements["ezkl_compile_circuit_time(s)"] = t

    start = time.perf_counter()
    get_srs(settings_path)
    t = time.perf_counter() - start
    print("GET_SRS took", t)
    perf_measurements["ezkl_get_srs_time(s)"] = t
    total_setup_time += t

    start = time.perf_counter()
    gen_witness(input_data_path, compiled_circuit_path, witness_path)
    t = time.perf_counter() - start
    print("GEN_WITNESS took", t)
    perf_measurements["ezkl_gen_witness_time(s)"] = t
    total_setup_time += t

    start = time.perf_counter()
    gen_keys(compiled_circuit_path, vk_path, pk_path)
    t = time.perf_counter() - start
    print("GEN_KEYS took", t)
    perf_measurements["ezkl_key_gen_time(s)"] = t
    total_setup_time += t
    perf_measurements["ezkl_setup_time(s)"] = total_setup_time

    if not setup_only:
        start = time.perf_counter()
        compute_proof(witness_path, compiled_circuit_path, pk_path, proof_path)
        t = time.perf_counter() - start
        print("PROVED took", t)
        perf_measurements["ezkl_proof_time(s)"] = t

    return perf_measurements


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    base_path = "ezkl_tmp"
    os.makedirs(base_path, exist_ok=True)

    try:
        input_data_path = "examples/onnx/bert/bert_input.json"
        onnx_model_path = "examples/onnx/bert/bert_tiny_squad.onnx"
        settings_path = os.path.join(base_path, "settings.json")
        compiled_circuit_path = os.path.join(base_path, "circuit.json")
        witness_path = os.path.join(base_path, "witness.json")
        vk_path = os.path.join(base_path, "vk.json")
        pk_path = os.path.join(base_path, "pk.json")
        proof_path = os.path.join(base_path, "proof.json")

        metrics = run_proof(
            onnx_model_path,
            input_data_path,
            settings_path,
            compiled_circuit_path,
            witness_path,
            vk_path,
            pk_path,
            proof_path,
        )

        print("Perf measurements:", metrics)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Optional: clean up
        # import shutil
        # shutil.rmtree(base_path)
        pass
