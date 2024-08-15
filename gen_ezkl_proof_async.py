import logging
import ezkl
import asyncio
import os
import time  # Import time module for timing
import numpy as np
import json

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Define paths
directory = 'examples/onnx/mnist_classifier'
model_path = os.path.join(directory, 'network.onnx')
data_path = os.path.join(directory, 'input.json')
compiled_model_path = os.path.join(directory, 'network.compiled')
pk_path = os.path.join(directory, 'key.pk')
vk_path = os.path.join(directory, 'key.vk')
settings_path = os.path.join(directory, 'settings.json')
witness_path = os.path.join(directory, 'witness.json')
cal_path = os.path.join(directory, 'calibration.json')
proof_path = os.path.join(directory, 'test.pf')

def time_function(func):
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

async def main():
    # Apply the time_function decorator
    gen_settings = time_function(ezkl.gen_settings)
    calibrate_settings = time_function(ezkl.calibrate_settings)
    compile_circuit = time_function(ezkl.compile_circuit)
    get_srs = time_function(ezkl.get_srs)
    gen_witness = time_function(ezkl.gen_witness)
    setup = time_function(ezkl.setup)
    prove = time_function(ezkl.prove)
    verify = time_function(ezkl.verify)

    # Time the gen_settings function
    gen_settings(model_path, settings_path)

    # Time the calibrate_settings function
    await calibrate_settings(data_path, model_path, settings_path, "resources")

    # Time the compile_circuit function
    compile_circuit(model_path, compiled_model_path, settings_path)

    # Time the get_srs function
    await get_srs(settings_path)

    # Time the gen_witness function
    gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # Time the setup function
    setup(compiled_model_path, vk_path, pk_path)
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Time the prove function
    prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
    assert os.path.isfile(proof_path)

    # Time the verify function
    verify(proof_path, settings_path, vk_path)
    print("verified")

# Run the asynchronous main function
if __name__ == '__main__':
    asyncio.run(main())
