from split_model import run_inference_on_onnx_model, get_intermediate_outputs, split_onnx_model_at_every_node
from log_utils import ExperimentLogger, time_function, print_func_exec_info, ResourceMonitor
from utils import count_onnx_model_parameters
import os
import ezkl

class EZKLProver:
    def __init__(self, worker_dir: str):
        self.directory = worker_dir
        self.model_path = os.path.join(self.directory, 'model.onnx')
        self.data_path = os.path.join(self.directory, 'input.json')
        self.compiled_model_path = os.path.join(self.directory, 'network.compiled')
        self.pk_path = os.path.join(self.directory, 'key.pk')
        self.vk_path = os.path.join(self.directory, 'key.vk')
        self.settings_path = os.path.join(self.directory, 'settings.json')
        self.witness_path = os.path.join(self.directory, 'witness.json')
        self.cal_path = os.path.join(self.directory, 'calibration.json')
        self.proof_path = os.path.join(self.directory, 'test.pf')
        self.exp_logger = ExperimentLogger(log_dir=self.directory)

    @time_function
    def gen_settings(self):
        assert ezkl.gen_settings(self.model_path, self.settings_path) == True

    @time_function
    def calibrate_settings(self):
        assert ezkl.calibrate_settings(self.data_path, self.model_path, self.settings_path, "resources") == True

    @time_function
    def compile_circuit(self):
        assert ezkl.compile_circuit(self.model_path, self.compiled_model_path, self.settings_path) == True

    @time_function
    def get_srs(self):
        ezkl.get_srs(self.settings_path)

    @time_function
    def gen_witness(self):
        ezkl.gen_witness(self.data_path, self.compiled_model_path, self.witness_path)
        assert os.path.isfile(self.witness_path)

    @time_function
    def setup(self):
        assert ezkl.setup(self.compiled_model_path, self.vk_path, self.pk_path) == True
        assert os.path.isfile(self.vk_path)
        assert os.path.isfile(self.pk_path)
        assert os.path.isfile(self.settings_path)

    @time_function
    def prove(self):
        ezkl.prove(self.witness_path, self.compiled_model_path, self.pk_path, self.proof_path, "single")
        assert os.path.isfile(self.proof_path)

    @time_function  
    def verify(self):
        res = ezkl.verify(self.proof_path, self.settings_path, self.vk_path)
        if res == True:
            print("verified")
        else:
            print("not verified")
    def run_end_to_end_proof(self):
        with ResourceMonitor() as monitor:
            num_parameters = count_onnx_model_parameters(self.model_path)
            # logging.info(f'Number of Model Parameters: {num_parameters}')
            self.exp_logger.log_value('num_model_params', num_parameters)
            self.exp_logger.log_env_resources()
            self.exp_logger.log_value('name', 'report')

            functions = [
                ('gen_settings', self.gen_settings),
                # ('calibrate_settings', self.calibrate_settings),
                ('compile_circuit', self.compile_circuit),
                ('get_srs', self.get_srs),
                ('gen_witness', self.gen_witness),
                ('setup', self.setup),
                ('prove', self.prove),
                ('verify', self.verify)
            ]
            
            for func_name, func in functions:
                execution_time = func()
                print_func_exec_info(func_name, execution_time)
                self.exp_logger.log_value(f'{func_name}(s)', execution_time)
            
            # Log resource data
            resource_data = monitor.resource_data
            self.exp_logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
            self.exp_logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
            self.exp_logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
            self.exp_logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
            self.exp_logger.flush_log()
            return self.proof_path


if __name__ == "__main__":

    models_to_test = [
        # ('examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx', 'examples/onnx/mobilenet/input.json'),
        ('examples/onnx/mnist_gan/network.onnx', 'examples/onnx/mnist_gan/input.json')]
    for onnx_file, input_file in models_to_test:

        # Get the output tensor(s) of every node in the model during inference
        intermediate_results = get_intermediate_outputs(onnx_file, input_file)

        models_with_inputs = split_onnx_model_at_every_node(onnx_file, input_file,  intermediate_results, 'examples/split_models/mnist_gan')  

        for idx, (sub_model_path, input_data_path) in enumerate(models_with_inputs):

            print(f'proving split {idx}')
            prover = EZKLProver(os.path.dirname(sub_model_path))
            proof_path = prover.run_end_to_end_proof()

