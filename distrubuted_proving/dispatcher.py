import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import logging
from utils import split_onnx_model, get_model_splits_inputs, get_num_parameters
import numpy as np
from onnx import shape_inference, ModelProto
from typing import List
import threading
import os
import json

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")
from concurrent.futures import ThreadPoolExecutor, as_completed

class Worker():   
    def __init__(self, id: int, address):
        self.id = id
        self.address = address
        self.is_free = True
        self.channel = None
        self.current_split = None

class ZKPProver():
    def __init__(self, config: DictConfig):
        self.workers:List[Worker] = []
        self.confirm_connections(config.worker_addresses)
        self.number_workers = len(self.workers) #plus one becuase this node also acts as a worker
    
    def confirm_connections(self, worker_addresses):
        for worker_address in worker_addresses:
            try:
                with grpc.insecure_channel(worker_address) as channel:
                    stub = pb2_grpc.WorkerStub(channel)
                    response = stub.Ping(pb2.Message(message='dispatcher'))
                    if response.received:
                        self.workers.append(Worker(id=len(self.workers), address=worker_address))
                        logger.info(f'Worker registered: {worker_address}')
                    else:
                        raise ConnectionError(f"Cannot connect to worker at {worker_address}") 
            except:
                raise ConnectionError(f"Cannot connect to worker at {worker_address}")
    

    def next_available_worker(self, split_idx):
        while True:
            for worker in self.workers:
                if worker.is_free:
                    return worker
            # Log a message if all workers are busy
            logging.info(f"All workers are busy. Waiting for a worker to become available to process {split_idx}.")
            # Add a short delay before checking again to avoid busy-waiting
            time.sleep(5)  # Adjust th


    def generate_proof(self, onnx_file, input_file):
        split_models = split_onnx_model(onnx_file, num_splits=len(self.workers))
        split_inputs = get_model_splits_inputs(split_models, input_file)
        # Combine items into tuples
        model_data_splits = list(zip(split_models, split_inputs))
        total_splits = len(model_data_splits)
        # Start a separate thread to monitor the workers' status
        monitor_thread = threading.Thread(target=self.monitor_workers, args=(total_splits,))
        monitor_thread.start()  

        for idx, (model, model_input) in enumerate(model_data_splits):
            worker:Worker = self.next_available_worker(idx)
            worker.channel = grpc.insecure_channel(worker.address)
            worker_response = self.send_grpc_request(worker, model, model_input)
            if 'computation started' in worker_response:
                logging.info(f"Started processing split {idx+1} on worker {worker.address}")
                worker.current_split = idx
                worker.is_free  = False 

        logging.info("All splits have been dispatched.")
        # Wait for the monitoring thread to finish
        monitor_thread.join()
        logging.info("All splits have been processed. Job done. Shutting Down")
        
    def send_grpc_request(self, worker: Worker, onnx_model:ModelProto, model_input):

        try:
            model_bytes = onnx_model.SerializeToString()
            data_array = np.array(model_input)
            reshaped_array = data_array.reshape(-1)
            data_list = reshaped_array.tolist()
            model_input = dict(input_data=[data_list])
            stub = pb2_grpc.WorkerStub(worker.channel)
            worker_response = stub.ComputeProof(pb2.ProofData(model_bytes=model_bytes, model_input=json.dumps(model_input)))
            return worker_response.message
        except Exception as e:
            print(f"Error occurred while sending gRPC request to worker {worker.address}: {e}")
    

    
    def monitor_workers(self, total_splits):
        splits_processed = []
        while len(splits_processed) < total_splits:
            time.sleep(5)
            for worker in self.workers:
                if not worker.is_free:

                    stub = pb2_grpc.WorkerStub(worker.channel)
                    response = stub.GetWorkerStatus(pb2.Message(message="status check"))
                    logging.info(f"Worker: {worker.address}, Status: {response.message}, Is Busy: {response.isbusy}, Processing Split: {worker.current_split}")

                    worker_busy = response.isbusy  # Replace this with actual check
                    if not worker_busy: #worker finished
                        splits_processed.append(worker.current_split)
                        worker.is_free = True
                        worker.current_split = None     

    # def send_grpc_request(self, worker: Worker, onnx_model:ModelProto, model_input):
    #     try:
    #         model_bytes = onnx_model.SerializeToString()
    #         data_array = np.array(model_input)
    #         reshaped_array = data_array.reshape(-1)
    #         data_list = reshaped_array.tolist()
    #         model_input = dict(input_data=[data_list])
    #         stub = pb2_grpc.WorkerStub(channel)
    #         future = stub.ComputeProof.future(pb2.ProofData(model_bytes=model_bytes, model_input=json.dumps(model_input)))
    #         return future

    #     except Exception as e:
    #         print(f"Error occurred while sending gRPC request to worker {worker.address}: {e}")
    

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    
    if not os.path.exists(config.model.onnx_file):
            raise FileNotFoundError(f"The specified file '{config.model.onnx_file}' does not exist.")
    
    if not os.path.exists(config.model.input_file):
            raise FileNotFoundError(f"The specified file '{config.model.input_file}' does not exist.")  
    
    print(f'model path: {config.model.onnx_file}')
    print(f'model input: {config.model.input_file}')
    print(f'# model parameters: {get_num_parameters(model_path=config.model.onnx_file)}')
    prover = ZKPProver(config=config)
    proof = prover.generate_proof(config.model.onnx_file, config.model.input_file)


if __name__ == '__main__':
    main()
