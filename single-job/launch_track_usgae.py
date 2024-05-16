import hydra
from omegaconf import DictConfig
import os 
import little_transformerr
import mnist_classifier
import nano_gpt
import logging
import torch
import json
import ezkl
import time
from functools import wraps
import psutil

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.DEBUG)

def time_and_resource_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the start time and initial CPU and memory usage
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_cpu_times = process.cpu_times()
        start_memory_info = process.memory_info()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get the end time and final CPU and memory usage
        end_time = time.time()
        end_cpu_times = process.cpu_times()
        end_memory_info = process.memory_info()

        # Calculate the execution time
        execution_time = end_time - start_time

        # Calculate CPU usage
        user_cpu_time = end_cpu_times.user - start_cpu_times.user
        system_cpu_time = end_cpu_times.system - start_cpu_times.system
        total_cpu_time = user_cpu_time + system_cpu_time

        # Calculate memory usage
        memory_usage = end_memory_info.rss - start_memory_info.rss

        # Print the execution time, CPU usage, and memory usage
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        print(f"CPU time: User {user_cpu_time:.4f} s, System {system_cpu_time:.4f} s, Total {total_cpu_time:.4f} s")
        print(f"Memory usage: {memory_usage / (1024 * 1024):.2f} MB")
        
        # Return the result of the function
        return result
    return wrapper

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    config.output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir,'network.onnx')
    compiled_model_path = os.path.join(config.output_dir,'network.compiled')
    pk_path = os.path.join(config.output_dir,'key.pk')
    vk_path = os.path.join(config.output_dir,'key.vk')
    settings_path = os.path.join(config.output_dir,'settings.json')
    witness_path = os.path.join(config.output_dir,'witness.json')
    data_path = os.path.join(config.output_dir,'input.json')
    cal_path = os.path.join(config.output_dir,"calibration.json")
    proof_path = os.path.join(config.output_dir,'test.pf')

    if config.model.name == 'little_transformer':
        model, shape, data_point =  little_transformerr.get_model(
            seq_len=config.model.seq_len,
            block_size=config.model.block_size,
            max_epochs=config.model.max_epochs,
            max_value=config.model.max_value,
            num_layers=config.model.num_layers,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.n_head,
            ff_dim=config.model.ff_dim)
    elif config.model.name == 'mnist_classifier':
        model, shape, data_point =  mnist_classifier.get_model()
    elif config.model.name == 'nano_gpt':
        model, shape, data_point =  nano_gpt.get_model(
            num_layers=config.model.num_layers,
            block_size=config.model.block_size,
            vocab_size=config.model.vocab_size,
            n_head=config.model.n_head,
            n_embd=config.model.n_embd)

    export_model_and_data(model=model, x=data_point, model_path=model_path, data_path=data_path)

    gen_settings(model_path, settings_path)
    calibrate_settings(data_path, model_path, settings_path)
    compile_circuit(model_path, compiled_model_path, settings_path)
    get_srs(settings_path)
    gen_witness(data_path, compiled_model_path, witness_path)
    setup(compiled_model_path, vk_path, pk_path, settings_path)
    prove(witness_path, compiled_model_path, pk_path, proof_path)
    verify(proof_path, settings_path, vk_path)

@time_and_resource_function
def verify(proof_path, settings_path, vk_path):
    res = ezkl.verify(proof_path, settings_path, vk_path)
    assert res == True
    print("verified")

@time_and_resource_function
def prove(witness_path, compiled_model_path, pk_path, proof_path):
    res = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
    print(res)
    assert os.path.isfile(proof_path)

@time_and_resource_function
def setup(compiled_model_path, vk_path, pk_path, settings_path):
    res = ezkl.setup(compiled_model_path, vk_path, pk_path)
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

@time_and_resource_function
def gen_witness(data_path, compiled_model_path, witness_path):
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

@time_and_resource_function
def get_srs(settings_path):
    res = ezkl.get_srs(settings_path)

@time_and_resource_function
def compile_circuit(model_path, compiled_model_path, settings_path):
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

@time_and_resource_function
def calibrate_settings(data_path, model_path, settings_path):
    res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
    assert res == True

@time_and_resource_function
def gen_settings(model_path, settings_path):
    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

@time_and_resource_function
def gen_iput_data(shape, cal_path):
    data_array = (torch.randn(20, *shape).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    json.dump(data, open(cal_path, 'w'))
    return data

@time_and_resource_function
def export_model_and_data(model, x, model_path, data_path):
    print(x)
    model.eval()
    model.to('cpu')
    torch.onnx.export(model, x, model_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    data_array = x.detach().numpy().reshape([-1]).tolist()
    data_json = dict(input_data=[data_array])
    print(data_json)
    json.dump(data_json, open(data_path, 'w'))

if __name__ == "__main__":
    main()
