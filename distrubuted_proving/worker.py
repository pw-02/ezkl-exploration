import grpc
import grpc
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import time
import threading
from concurrent import futures
import logging
from datetime import datetime
import ezkl
# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import os
from log_utils import ExperimentLogger, time_function, print_func_exec_info
from utils import get_num_parameters
import datetime
import onnx
from utils import split_onnx_model, get_model_splits_inputs, get_num_parameters
from onnx import ModelProto

class EZKLProver():
    def __init__(self, worker_dir:str):
        self.directory = worker_dir
        self.model_path =  os.path.join(self.directory, f'model.onnx')
        self.data_path = os.path.join(self.directory, f'input.json')
        self.compiled_model_path = os.path.join(self.directory, f'network.compiled')
        self.pk_path = os.path.join(self.directory, f'key.pk')
        self.vk_path = os.path.join(self.directory, f'key.vk')
        self.settings_path = os.path.join(self.directory, f'settings.json')
        self.witness_path = os.path.join(self.directory, f'witness.json')
        self.cal_path = os.path.join(self.directory, f'calibration.json')
        self.proof_path = os.path.join(self.directory, f'test.pf')
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
        
            num_parameters =get_num_parameters(self.model_path)
            print(f'Number Model Parmeters: {num_parameters}')
            self.exp_logger.log_value('num_model_params', num_parameters)
            self.exp_logger.log_env_resources()
            self.exp_logger.log_value('name', 'report')

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
                    print_func_exec_info(func_name, execution_time)
                    self.exp_logger.log_value(f'{func_name}(s)', execution_time)
            
                  # Log resource data
            # resource_data = monitor.resource_data
            # self.exp_logger.log_value('mean_cpu', resource_data["cpu_util"]["mean"])
            # self.exp_logger.log_value('max_cpu', resource_data["cpu_util"]["max"])
            # self.exp_logger.log_value('mean_cpu_mem_gb', resource_data["cpu_mem_gb"]["mean"])
            # self.exp_logger.log_value('max_cpu_mem_gb', resource_data["cpu_mem_gb"]["max"])
            self.exp_logger.flush_log()
            

class WorkerServicer(pb2_grpc.WorkerServicer):

    def ProcessTask(self, request, context):
        logging.info("Received task: %s", request)
        # Process the task
        # Simulating processing time
        time.sleep(2)
        # Return response
        return pb2.Task(id=request.id, data="Processed " + request.data)
    
    def ComputeProof(self, request, context):
        logging.info("Received 'Compute Proof' task")

         # Define a function to compute the proof
        def compute_proof():
            import numpy as np
            import json

            # Format the date and time as a string
            directory_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            directory_path = os.path.join("data", directory_name)
            os.makedirs(directory_path, exist_ok = True)

            received_model = ModelProto()
            received_model.ParseFromString(request.model_bytes)
            onnx.save(received_model, os.path.join(directory_path, f'model.onnx'))
            model_input = json.loads(request.model_input)

            json.dump(model_input, open(os.path.join(directory_path, f'input.json'), 'w'))

            prover = EZKLProver(directory_path)
            prover.run_end_to_end_proof()
            logging.info("Proof computed and verified")

            # # Once the proof is computed, update the response
            # result = {'message': 'Proof computed and verified'}
            # context.set_details(pb2.Message(**result))

        # Start a new thread to compute the proof
        proof_thread = threading.Thread(target=compute_proof)
        proof_thread.start()

        # Return the initial response immediately
        return pb2.Message(message="Proof computation started")


    
    def Ping(self, request, context): 
        logging.info("Received Ping Request from %s", request.message)
        # Return response
        result = {'message': 'pong', 'received': True}
        return pb2.MessageResponse(**result)
    


def run_worker(port):
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pb2_grpc.add_WorkerServicer_to_server(WorkerServicer(), server)
        server.add_insecure_port('[::]:' + str(port))
        server.start()
        logging.info("Worker started...")
        # # Register with dispatcher
        # channel = grpc.insecure_channel("localhost:50051")
        # stub = pb2_grpc.DispatcherStub(channel)
        # response = stub.RegisterWorker(pb2.WorkerAddress(address="localhost:50052"))  # Change the address as needed
        # print("Registration response:", response)
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
        logging.exception(f"Error in serve(): {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run gRPC worker")
    parser.add_argument("--port", type=int, default=50052, help="Port number for the worker to listen on")
    args = parser.parse_args()
    run_worker(args.port)
