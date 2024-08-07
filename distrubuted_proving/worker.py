import grpc
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import time
import threading
from concurrent import futures
import logging
from datetime import datetime
import ezkl
import os
import json
import onnx
from onnx import ModelProto
from log_utils import ExperimentLogger, time_function, print_func_exec_info, ResourceMonitor
from utils import count_onnx_model_operations

# # Configure logging
# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# # Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPWorker")

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
        assert ezkl.verify(self.proof_path, self.settings_path, self.vk_path) == True

    def run_end_to_end_proof(self):
        with ResourceMonitor() as monitor:
            num_parameters = count_onnx_model_operations(self.model_path)
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
                logger.info(f'{func_name} took {execution_time} seconds')
                print_func_exec_info(func_name, execution_time)
                self.exp_logger.log_value(f'{func_name}(s)', execution_time)
            
            resource_data = monitor.resource_data
            self.exp_logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
            self.exp_logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
            self.exp_logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
            self.exp_logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
            self.exp_logger.flush_log()
            return self.proof_path

class ZKPWorkerServicer(pb2_grpc.ZKPWorkerServiceServicer):
    def __init__(self):
        self.status = 'awaiting work'
        self.is_busy = False
        self.computed_proof = None

    def Ping(self, request, context):
        logging.info("Received Ping Request from %s", request.message)
        return pb2.MessageResponse(message='pong', received=True)

    def ComputeProof(self, request, context):
        logging.info("Received 'Compute Proof' request.")
        
        directory_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_path = os.path.join("data", directory_name)
        os.makedirs(directory_path, exist_ok=True)

        with open(os.path.join(directory_path, 'model.onnx'), 'wb') as f:
            f.write(request.onnx_model)

        model_input = json.loads(request.input_data)
        json.dump(model_input, open(os.path.join(directory_path, 'input.json'), 'w'))

        prover = EZKLProver(directory_path)
        proof_path = prover.run_end_to_end_proof()
            
        with open(proof_path, "rb") as file:
            self.computed_proof = file.read()

        logging.info("Proof computed and verified.")

        # proof_thread = threading.Thread(target=compute_proof)
        # proof_thread.start()
       
        return pb2.ProofResponse(success=True, proof=self.computed_proof)

def run_server(port):
    try:
        max_message_length = 2**31 - 1  # ~2GB

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=[
                ('grpc.max_send_message_length', max_message_length),
                ('grpc.max_receive_message_length', max_message_length),
            ]
        )
        pb2_grpc.add_ZKPWorkerServiceServicer_to_server(ZKPWorkerServicer(), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        logging.info(f"Worker started on port {port}...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
        logging.exception("Error in server: %s", e)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run gRPC server")
    parser.add_argument("--port", type=int, default=50052, help="Port number for the server to listen on")
    args = parser.parse_args()
    run_server(args.port)

if __name__ == '__main__':
    main()
