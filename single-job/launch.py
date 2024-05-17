import hydra
from omegaconf import DictConfig
import os
import logging
import torch
import json
import ezkl
import time
from functools import wraps
import psutil
from log_util import ExperimentLogger, ResourceMonitor
import little_transformerr
import mnist_classifier
import nano_gpt
# Configure logging
FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.DEBUG)

# Decorator to time functions
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return execution_time
    return wrapper

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    exp_logger = ExperimentLogger(log_dir=config.log_dir)

    # Log configuration
    for key, value in config.model.items():
        exp_logger.log_value(key, value)

    # Create output directory
    config.output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Define paths
    paths = {
        'model_path': os.path.join(config.output_dir, 'network.onnx'),
        'compiled_model_path': os.path.join(config.output_dir, 'network.compiled'),
        'pk_path': os.path.join(config.output_dir, 'key.pk'),
        'vk_path': os.path.join(config.output_dir, 'key.vk'),
        'settings_path': os.path.join(config.output_dir, 'settings.json'),
        'witness_path': os.path.join(config.output_dir, 'witness.json'),
        'data_path': os.path.join(config.output_dir, 'input.json'),
        'cal_path': os.path.join(config.output_dir, "calibration.json"),
        'proof_path': os.path.join(config.output_dir, 'test.pf')
    }

    # Get model based on configuration
    if config.model.name == 'little_transformer':
        model, shape, data_point, num_params = little_transformerr.get_model(
            seq_len=config.model.seq_len,
            block_size=config.model.block_size,
            max_epochs=config.model.max_epochs,
            max_value=config.model.max_value,
            num_layers=config.model.num_layers,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.n_head,
            ff_dim=config.model.ff_dim
        )
    elif config.model.name == 'mnist_classifier':
        model, shape, data_point, num_params = mnist_classifier.get_model()
    elif config.model.name == 'nano_gpt':
        model, shape, data_point, num_params = nano_gpt.get_model(
            num_layers=config.model.num_layers,
            block_size=config.model.block_size,
            vocab_size=config.model.vocab_size,
            n_head=config.model.n_head,
            n_embd=config.model.n_embd
        )

    exp_logger.log_value('num_model_params', num_params)
    exp_logger.log_env_resources()

    def print_func_exec_info(func_name: str, duration, monitor: ResourceMonitor):
        resource_data = monitor.resource_data
        mean_cpu = resource_data["cpu_util"]["mean"]
        max_cpu = resource_data["cpu_util"]["max"]
        mean_cpu_mem_gb = resource_data["cpu_mem_gb"]["mean"]
        max_cpu_mem_gb = resource_data["cpu_mem_gb"]["max"]
        print(f'Step: {func_name}\t Duration: {duration:.4f}s\t CPU(mean): {mean_cpu:.2f}%\t '
              f'CPU(max): {max_cpu:.2f}%\t Mem(mean): {mean_cpu_mem_gb:.2f}GB\t Mem(max): {max_cpu_mem_gb:.2f}GB')

    with ResourceMonitor() as monitor:
        export_model_and_data(model=model, x=data_point, model_path=paths['model_path'], data_path=paths['data_path'])
        for func_name, func in [
            ('gen_settings', gen_settings),
            ('calibrate_settings', calibrate_settings),
            ('compile_circuit', compile_circuit),
            ('get_srs', get_srs),
            ('gen_witness', gen_witness),
            ('setup', setup),
            ('prove', prove),
            ('verify', verify)
        ]:
            execution_time = func(paths)
            print_func_exec_info(func_name, execution_time, monitor)
            exp_logger.log_value(f'{func_name}(s)', execution_time)

        # Log resource data
        resource_data = monitor.resource_data
        exp_logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
        exp_logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
        exp_logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
        exp_logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
        exp_logger.flush_log()
    
@time_function
def gen_settings(paths):
    res = ezkl.gen_settings(paths['model_path'], paths['settings_path'])
    assert res == True

@time_function
def calibrate_settings(paths):
    res = ezkl.calibrate_settings(paths['data_path'], paths['model_path'], paths['settings_path'], "resources")
    assert res == True

@time_function
def compile_circuit(paths):
    res = ezkl.compile_circuit(paths['model_path'], paths['compiled_model_path'], paths['settings_path'])
    assert res == True

@time_function
def get_srs(paths):
    res = ezkl.get_srs(paths['settings_path'])

@time_function
def gen_witness(paths):
    res = ezkl.gen_witness(paths['data_path'], paths['compiled_model_path'], paths['witness_path'])
    assert os.path.isfile(paths['witness_path'])

@time_function
def setup(paths):
    res = ezkl.setup(paths['compiled_model_path'], paths['vk_path'], paths['pk_path'])
    assert res == True
    assert os.path.isfile(paths['vk_path'])
    assert os.path.isfile(paths['pk_path'])
    assert os.path.isfile(paths['settings_path'])

@time_function
def prove(paths):
    res = ezkl.prove(paths['witness_path'], paths['compiled_model_path'], paths['pk_path'], paths['proof_path'], "single")
    assert os.path.isfile(paths['proof_path'])

@time_function
def verify(paths):
    res = ezkl.verify(paths['proof_path'], paths['settings_path'], paths['vk_path'])
    assert res == True

@time_function
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
