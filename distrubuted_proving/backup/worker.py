import os
import ezkl
from utils import time_function
from log_utils import ExperimentLogger, ResourceMonitor, print_func_exec_info

class EZKLProver():
    def __init__(self, worker_id, worker_dir:str):
        self.directory = worker_dir
        self.id = worker_id
        self.model_path =  os.path.join(self.directory, f'model_{self.id}.onnx')
        self.data_path = os.path.join(self.directory, f'input_{self.id}.json')
        self.compiled_model_path = os.path.join(self.directory, f'network_{self.id}.compiled')
        self.pk_path = os.path.join(self.directory, f'key_{self.id}.pk')
        self.vk_path = os.path.join(self.directory, f'key_{self.id}.vk')
        self.settings_path = os.path.join(self.directory, f'settings_{self.id}.json')
        self.witness_path = os.path.join(self.directory, f'witness_{self.id}.json')
        self.cal_path = os.path.join(self.directory, f'calibration_{self.id}.json')
        self.proof_path = os.path.join(self.directory, f'test_{self.id}.pf')
        self.exp_logger = ExperimentLogger(log_dir=self.directory)

    @time_function
    def gen_settings(self):
        res = ezkl.gen_settings(self.model_path, self.settings_path)
        assert res == True
    @time_function
    def calibrate_settings(self):
        res = ezkl.calibrate_settings(self.data_path,self.model_path, self.settings_path, "resources")
        assert res == True
    @time_function
    def compile_circuit(self):
        res = ezkl.compile_circuit(self.model_path, self.compiled_model_path, self.settings_path)
        assert res == True

    @time_function
    def get_srs(self):
        res = ezkl.get_srs(self.settings_path)

    @time_function
    def gen_witness(self):
        res = ezkl.gen_witness(self.data_path, self.compiled_model_path, self.witness_path)
        assert os.path.isfile(self.witness_path)

    @time_function
    def setup(self): 
        res = ezkl.setup(self.compiled_model_path, self.vk_path, self.pk_path)
        assert res == True
        assert os.path.isfile(self.vk_path)
        assert os.path.isfile(self.pk_path)
        assert os.path.isfile(self.settings_path)

    @time_function
    def prove(self):
        res = ezkl.prove(self.witness_path,self.compiled_model_path,self.pk_path,self.proof_path, "single")
        assert os.path.isfile(self.proof_path)

    @time_function  
    def verify(self):
        res = ezkl.verify(self.proof_path,self.settings_path,self.vk_path)
        assert res == True
    

    def run_end_to_end_proof(self):
        #   with ResourceMonitor() as monitor:
            for func_name, func in [
                ('gen_settings', self.gen_settings),
                ('calibrate_settings', self.calibrate_settings),
                ('compile_circuit', self.compile_circuit),
                ('get_srs', self.get_srs),
                ('gen_witness', self.gen_witness),
                ('setup', self.setup),
                ('prove', self.prove),
                ('verify', self.verify)
                ]:
                    execution_time = func()
                    # print_func_exec_info(func_name, execution_time, monitor)
                    self.exp_logger.log_value(f'{func_name}(s)', execution_time)
            
                  # Log resource data
            # resource_data = monitor.resource_data
            # self.exp_logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
            # self.exp_logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
            # self.exp_logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
            # self.exp_logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
            self.exp_logger.flush_log()
            

    