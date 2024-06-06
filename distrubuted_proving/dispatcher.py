import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import os
import shutil
import queue
import logging
from datetime import datetime
from utils import split_onnx_model, get_model_splits_inputs, get_num_parameters
import onnx
import json
import numpy as np
from onnx import shape_inference, ModelProto

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")
from concurrent.futures import ThreadPoolExecutor, as_completed


class Worker():   
    def __init__(self, id: int, address, worker_dir):
        self.id = id
        self.address = address
        self.directory = worker_dir
        self.is_busy = False
        os.makedirs(self.directory, exist_ok=True)
    
class ZKPProver():
    def __init__(self, config: DictConfig):
        
        self.worker_addresses = config.worker_addresses
        self.working_dir = config.working_dir
        self.workers = {}

        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

        os.makedirs(self.working_dir, exist_ok=True)
        self.workers = {}
        self.confirm_connections()
        self.number_workers = len(self.workers) #plus one becuase this node also acts as a worker

    def confirm_connections(self):
        for worker_address in self.worker_addresses:
             # Forward task to the worker
            try:
                with grpc.insecure_channel(worker_address) as channel:
                    stub = pb2_grpc.WorkerStub(channel)
                    response = stub.Ping(pb2.Message(message='dispatcher'))
                    if response.received:
                        worker = Worker(id=len(self.workers), address=worker_address, worker_dir=os.path.join(self.working_dir, f'worker_{len(self.workers)}'))
                        self.workers[worker.id] = worker
                        logger.info(f'Worker registered:{worker_address}')
                    else:
                        raise ConnectionError(f"Cannot connect to worker at {worker_address}") 
            except:
                raise ConnectionError(f"Cannot connect to worker at {worker_address}") 

    def generate_proof(self, onnx_file, input_file):
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"The specified file '{onnx_file}' does not exist.")
        
        split_models = split_onnx_model(onnx_file, num_splits=len(self.workers))
        split_inputs = get_model_splits_inputs(split_models, input_file)
        futures = []
        channels = []
        
        for idx, worker in self.workers.items():
            channel = grpc.insecure_channel(worker.address)
            future = self.send_grpc_request(channel, worker, split_models[idx], split_inputs[idx], onnx_file=onnx_file, data_path=input_file)
            futures.append(future)
            channels.append(channel)  # Keep a reference to the channel to prevent it from being closed

        # Wait for all futures to complete
        for future in futures:
            future.result()  # This will block until the future is done

        for channel in channels:
            channel.close()  # Explicitly close the channel when done
      
        print("All done") 

    def send_grpc_request(self, channel, worker: Worker, onnx_model:ModelProto, model_input, onnx_file, data_path):
        try:
            
            model_bytes = onnx_model.SerializeToString()
            data_array = np.array(model_input)
            reshaped_array = data_array.reshape(-1)
            data_list = reshaped_array.tolist()
            model_input = dict(input_data=[data_list])
            stub = pb2_grpc.WorkerStub(channel)
            future = stub.ComputeProof.future(pb2.ProofData(model_bytes=model_bytes, model_input=json.dumps(model_input)))
            return future

        except Exception as e:
            print(f"Error occurred while sending gRPC request to worker {worker.address}: {e}")
    

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    print(f'model path: {config.model.onnx_file}')
    print(f'model input: {config.model.input_file}')
    print(f'# model parameters: {get_num_parameters(model_path=config.model.onnx_file)}')
    prover = ZKPProver(config=config)
    proof = prover.generate_proof(config.model.onnx_file, config.model.input_file)


if __name__ == '__main__':
    main()
