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
import onnx
from grpc import Channel 
# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")
from concurrent.futures import ThreadPoolExecutor, as_completed

class Worker():   
    def __init__(self, id: int, address):
        self.id = id
        self.address = address
        self.is_free = True
        self.channel:Channel = None
        self.current_model_part: ModelPart = None

class ModelPart():   
    def __init__(self, idx: int):
         self.part_idx = idx
         self.computed_witness = None
         self.computed_proof = None
         self.is_completed = False

class ZKPProver():
    def __init__(self, config: DictConfig):
        self.workers:List[Worker] = []
        self.confirm_connections(config.worker_addresses)
        self.number_of_workers = len(self.workers) #plus one becuase this node also acts as a worker
        self.model_parts:List[ModelPart] = []
        
    def confirm_connections(self, worker_addresses):
        max_message_length = 2**31 - 1  # This is 2,147,483,647 bytes (~2GB)

        for worker_address in worker_addresses:
            try:
                with grpc.insecure_channel(
                    worker_address,
                    options=[
                        ('grpc.max_send_message_length', max_message_length),
                        ('grpc.max_receive_message_length', max_message_length),
                    ]
                ) as channel:
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
            logging.info(f"All workers are busy. Waiting for a worker to become available to process model part {split_idx}")
            # Add a short delay before checking again to avoid busy-waiting
            time.sleep(30)  # Adjust th
    
    def generate_proof(self, onnx_file, input_file,n_parts = None ):

        if not n_parts:
            n_parts = self.number_of_workers

        split_models = split_onnx_model(onnx_file, n_parts=n_parts, max_parameters_threshold=208067)
        split_inputs = get_model_splits_inputs(split_models, input_file)
        model_data_splits = list(zip(split_models, split_inputs))
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_progress, args=(len(model_data_splits),))
        monitor_thread.start()

        for idx, (model, model_input) in enumerate(model_data_splits):

            self.process_split(idx, model, model_input)

        logging.info("All splits have been dispatched.")
        # Wait for the monitoring thread to finish
        monitor_thread.join()
        logging.info("All splits have been processed. Job done. Shutting Down")
    
    def process_split(self, idx, model, model_input):
        model_part = ModelPart(idx=len(self.model_parts)+1)
        self.model_parts.append(model_part)
        worker = self.next_available_worker(model_part.part_idx)
        worker.channel = grpc.insecure_channel(worker.address)
        worker_response = self.send_grpc_request(worker, model, model_input)

        if 'computation started' in worker_response:
            logging.info(f"Started processing model part {model_part.part_idx} on worker {worker.address}")
            worker.current_model_part = model_part
            worker.is_free = False

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
    
    def monitor_progress(self, total_parts):
        time.sleep(10)  # Check every 5 seconds
        while True:  # Continuously monitor
            all_proofs_ready = all(part.computed_proof is not None for part in self.model_parts)

            if all_proofs_ready and len(self.model_parts) == total_parts:
                logging.info("All proofs are computed. Exiting monitor_progress.")
                break  # Exit the loop if all proofs are computed

            time.sleep(5)  # Check every 5 seconds

            for worker in self.workers:
                if not worker.is_free:
                    # Check if the channel is already closed or in an error state
                    if worker.channel is None:
                        worker.channel = grpc.insecure_channel(worker.address)
                    stub = pb2_grpc.WorkerStub(worker.channel)
                    try:
                        response = stub.GetComputedProof(pb2.Message(message="proof status check"))
                        status_message = response.status_message
                        proof = response.proof
                        if len(proof) > 1:
                            logging.info(f"Worker: {worker.address} | Model part: {worker.current_model_part.part_idx} | {status_message} | Proof Size: {len(proof)}")
                            # Update the corresponding ModelPart object
                            worker.current_model_part.computed_proof = proof
                            worker.current_model_part = None
                            worker.is_free = True
                        else:
                            logging.info(f"Worker: {worker.address} | Model part: {worker.current_model_part.part_idx} | {status_message}")
                    except grpc.RpcError as e:
                        logging.error(f"Error occurred while communicating with worker: {worker.address}. Error: {e}")
    

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
    prover.generate_proof(config.model.onnx_file, config.model.input_file, config.num_model_parts)


if __name__ == '__main__':
    main()



   # def generate_proof(self, onnx_file, input_file):
    #     # split_models = split_onnx_model(onnx_file, num_splits=self.number_workers)
    #     split_models = split_onnx_model(onnx_file, n_parts=2)
    #     split_inputs = get_model_splits_inputs(split_models, input_file)
    #     model_data_splits = list(zip(split_models, split_inputs))
        
    #     monitor_thread = threading.Thread(target=self.monitor_progress)
    #     monitor_thread.start()  

    #     for idx, (model, model_input) in enumerate(model_data_splits):
    #         self.model_parts.append(ModelPart(idx=idx))

    #     for idx, (model, model_input) in enumerate(model_data_splits):
    #         worker:Worker = self.next_available_worker(idx)
    #         worker.channel = grpc.insecure_channel(worker.address)
    #         worker_response = self.send_grpc_request(worker, model, model_input)
    #         if 'computation started' in worker_response:
    #             logging.info(f"Started processing split {idx+1} on worker {worker.address}")
    #             worker.current_split = idx
    #             worker.is_free  = False 

    #     logging.info("All splits have been dispatched.")
    #     # Wait for the monitoring thread to finish
    #     monitor_thread.join()
    #     logging.info("All splits have been processed. Job done. Shutting Down")