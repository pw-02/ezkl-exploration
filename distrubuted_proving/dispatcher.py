import grpc
import time
import proto.zkpservice_pb2_grpc as pb2_grpc
import proto.zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import os
import shutil
import queue
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")

class ZKPProver():
    def __init__(self, config: DictConfig):

        self.worker_addresses = config.worker_addresses
        self.working_dir = config.working_dir
        self.worker_queue = queue.Queue()
        # self.worker_queue.put('local')
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

        os.makedirs(self.working_dir, exist_ok=True)

        self.confirm_connections()
        self.number_workers = self.worker_queue.qsize() + 1 #plus one becuase this node also acts as a worker


    def confirm_connections(self):
        for worker_address in self.worker_addresses:
             # Forward task to the worker
            try:
                with grpc.insecure_channel(worker_address) as channel:
                    stub = pb2_grpc.WorkerStub(channel)
                    response = stub.Ping(pb2.Message(message='dispatcher'))
                    if response.received:
                        logger.info(f'Worker registered:{worker_address}')
                        self.worker_queue.put(worker_address)
                    else:
                        raise ConnectionError(f"Cannot connect to worker at {worker_address}") 
            except:
                raise ConnectionError(f"Cannot connect to worker at {worker_address}") 
            
    def generate_proof(self, onnx_file, input_file):
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"The specified file '{onnx_file}' does not exist.")
        
        



        pass

        #split the model evenly amound all workers
        

    
    
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
