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
from utils import split_onnx_model, get_model_splits_inputs
import onnx
import json
import torch
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
            
        split_models = split_onnx_model(onnx_file, num_splits=2)
        split_inputs = get_model_splits_inputs(split_models, input_file)
        
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = {executor.submit(self.send_grpc_request, worker, split_models[idx], split_inputs[idx]): worker for idx, worker in self.workers.items()}
            
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    response = future.result()
                    print(f"Received response from worker {worker.address}: {response}")
                except Exception as exc:
                    print(f"Worker {worker.address} generated an exception: {exc}")

    
    def send_grpc_request(self, worker:Worker, split_model, split_input):
        try:
            onnx.save(split_model, os.path.join(worker.directory, f'model.onnx'))
            data_tensor = torch.tensor(split_input)
            data_array = data_tensor.detach().numpy().reshape([-1]).tolist()
            data_json = dict(input_data=[data_array])
            json.dump(data_json, open(os.path.join(worker.directory, f'input.json'), 'w'))

            with grpc.insecure_channel(worker.address) as channel:
                stub = pb2_grpc.WorkerStub(channel)
                response = stub.ComputeProof(pb2.ProofInfo(model_path=worker.directory, data_path=True))
                return response
        except Exception as e:
            print(f"Error occurred while sending gRPC request to worker {worker.address}: {e}")
   
    # def generate_proof(self, onnx_file, input_file):
    #     if not os.path.exists(onnx_file):
    #         raise FileNotFoundError(f"The specified file '{onnx_file}' does not exist.")
            
    #     split_models = split_onnx_model(onnx_file, num_splits=2)
    #     split_inputs = get_model_splits_inputs(split_models, input_file)
        
    #     for idx, split in enumerate(split_models):
    #         #save_model_split
    #         worker:Worker = self.workers[idx]
    #         onnx.save(split, os.path.join(worker.directory, f'model.onnx'))
    #         data_tensor = torch.tensor(split_inputs[idx])
    #         data_array = data_tensor.detach().numpy().reshape([-1]).tolist()
    #         data_json = dict(input_data=[data_array])
    #         json.dump(data_json, open(os.path.join(worker.directory, f'input.json'), 'w'))
        
    #         # Execute gRPC requests concurrently
    #     with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
    #         futures = {executor.submit(self.send_grpc_request, worker): worker for worker in self.workers}
            
    #         for future in as_completed(futures):
    #             worker = futures[future]
    #             try:
    #                 response = future.result()
    #                 print(f"Received response from worker {worker.address}: {response}")
    #             except Exception as exc:
    #                 print(f"Worker {worker.address} generated an exception: {exc}")


    # def send_grpc_request(self, worker):
    #     with grpc.insecure_channel(worker.address) as channel:
    #         stub = pb2_grpc.WorkerStub(channel)
    #         response = stub.ComputeProof(pb2.ProofInfo(model_path=worker.model_path, data_path=worker.data_path))
    #         return response




    # def Dispatch(self):
    #     # Here goes the logic to dispatch tasks to workers
    #     print("Received task:", request)
    #     try:
    #         worker_address = self.worker_queue.get(timeout=5)  # Timeout after 5 seconds
    #     except queue.Empty:
    #         print("No available workers.")
    #         return pb2.Task(id="", data="No available workers.")
        
    #     # Forward task to the worker
    #     with grpc.insecure_channel(worker_address) as channel:
    #         stub = pb2_grpc.WorkerStub(channel)
    #         response = stub.ProcessTask(request)
    #         print("Received response from worker:", response)
    #         return response



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    prover = ZKPProver(config=config)
    proof = prover.generate_proof(config.model.onnx_file, config.model.input_file)


if __name__ == '__main__':
    main()
