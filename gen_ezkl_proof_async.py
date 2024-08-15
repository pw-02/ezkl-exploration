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
directory = 'ezkl_python/mnist_classifier'
model_path = os.path.join(directory, 'network.onnx')
data_path = os.path.join(directory, 'input.json')
compiled_model_path = os.path.join(directory, 'network.compiled')
pk_path = os.path.join(directory, 'key.pk')
vk_path = os.path.join(directory, 'key.vk')
settings_path = os.path.join(directory, 'settings.json')
witness_path = os.path.join(directory, 'witness.json')
cal_path = os.path.join(directory, 'calibration.json')
proof_path = os.path.join(directory, 'test.pf')

async def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = await func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{func.__name__} took {execution_time:.2f} seconds")
    return result

async def main():
    # Time the gen_settings function
    await time_function(ezkl.gen_settings, model_path, settings_path)

    # Time the calibrate_settings function
    await time_function(ezkl.calibrate_settings, data_path, model_path, settings_path, "resources", scales=[2,7])

    # Time the compile_circuit function
    await time_function(ezkl.compile_circuit, model_path, compiled_model_path, settings_path)

    # Time the get_srs function
    await time_function(ezkl.get_srs, settings_path)

    # Time the gen_witness function
    await time_function(ezkl.gen_witness, data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # Time the setup function
    await time_function(ezkl.setup, compiled_model_path, vk_path, pk_path)
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Time the prove function
    await time_function(ezkl.prove, witness_path, compiled_model_path, pk_path, proof_path, "single")
    assert os.path.isfile(proof_path)

    # Time the verify function
    await time_function(ezkl.verify, proof_path, settings_path, vk_path)
    print("verified")

# Run the asynchronous main function
if __name__ == '__main__':
    asyncio.run(main())
