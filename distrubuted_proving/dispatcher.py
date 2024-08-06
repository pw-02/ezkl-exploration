import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import logging
from utils import  analyze_onnx_model_for_zk_proving, load_onnx_model
from split_model import get_intermediate_outputs, split_onnx_model_at_every_node
from typing import List
import os
from queue import Queue
from grpc import Channel 
from log_utils import ResourceMonitor
from worker import EZKLProver
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
        self.channel:Channel = None
        self.assigned_model_part: OnnxModel = None

class OnnxModel ():   
    def __init__(self, id: int, onnx_model_path:str, input_data:str, model_bytes = None):
         self.id = id
        #  self.model_bytes = onnx_model.SerializeToString()
         self.onnx_model_path = onnx_model_path
         self.computed_witness = None
         self.computed_proof = None
         self.is_completed = False
         self.input_data = input_data
         self.sub_models: List[OnnxModel] = []
         self.model_bytes = model_bytes
         self.info = analyze_onnx_model_for_zk_proving(onnx_model_path=onnx_model_path)


class ZKPProver():
    def __init__(self, config: DictConfig):
        self.workers:List[Worker] = []
        self.check_worker_connections(config.worker_addresses)
        self.number_of_workers = len(self.workers)
        # self.model_parts:List[SubModel] = []
    
    def prepare_model_for_distrubuted_proving(self, onnx_model_path:str, json_input_file:str):

        #get the output tensor(s) of every node node in the model during inference
        global_model = OnnxModel('1', onnx_model_path, json_input_file)
        
        logger.info(f'Num model params: {global_model.info["num_model_params"]}, Num rows in zk circuit: {global_model.info["zk_circuit_num_rows"]}, Number of nodes: {global_model.info["num_model_ops"]}')
        logger.info(f'Splitting model at every node..')
        node_outputs = get_intermediate_outputs(onnx_model_path, json_input_file)
        sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input_file, node_outputs, 'tmp')

        #add in some logic here later if we need to combine split models for load balancing

        for idx, (sub_model_path, input_data) in enumerate(sub_models):
            sub_model = OnnxModel(f'{str(global_model.id)}.{str(idx+1)}', sub_model_path,input_data)
            global_model.sub_models.append(sub_model)
        
        self.compute_proof(global_model)
    
    def compute_proof(self, onnx_model: OnnxModel):
        import datetime
        import shutil
        """
        Computes zero-knowledge proofs for the given ONNX model by distributing
        sub-models to available workers.
        """

        logger.info(f'Starting proof computation for every sub model..')
        self.workers.append( Worker(id=1, address='local'))

        def send_proof_request(worker: Worker, sub_model: OnnxModel):
            """
            Sends a proof computation request to a specific worker.
            """
            try:
                # onnx_model = load_onnx_model(sub_model.onnx_model_path) 
                # model_bytes = onnx_model.SerializeToString()
                directory_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                directory_path = os.path.join("worker_data", directory_name)
                os.makedirs(directory_path, exist_ok = True)
                # Copy the ONNX model file to the new directory
                shutil.copy(sub_model.onnx_model_path, os.path.join(directory_path, f'model.onnx'))
                input_dict = {"input_data": sub_model.input_data}
                json.dump(input_dict, open(os.path.join(directory_path, f'input.json'), 'w'))
                prover = EZKLProver(directory_path)
                proof_path = prover.run_end_to_end_proof()
                with open(proof_path, "rb") as file:
                    sub_model.computed_proof = file.read()
                
                sub_model.is_completed = True
                logger.info(f'Proof computed for sub-model {sub_model.id}')
            except Exception as e:
                logger.error(f'Exception occurred while computing proof for sub-model {sub_model.id}: {e}')
            finally:
                worker.is_free = True  # Mark worker as free once done

        # Initialize the task queue with sub-models
        task_queue = Queue()
        for sub_model in onnx_model.sub_models:
            task_queue.put(sub_model)

        # Define a function to process tasks
        def process_tasks():
            while not task_queue.empty():
                # Find a free worker
                free_worker = None
                while free_worker is None:
                    for worker in self.workers:
                        if worker.is_free:
                            free_worker = worker
                            break
                    if free_worker is None:
                        time.sleep(10)  # Wait for some time before checking again
                
                # Get the next sub-model from the queue
                sub_model = task_queue.get()
                free_worker.is_free = False
                # Submit the proof request task
                executor.submit(send_proof_request, free_worker, sub_model)
        
        # Create a ThreadPoolExecutor for handling tasks
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            # Start processing tasks
            process_tasks()

        # Wait for all tasks to complete
        executor.shutdown(wait=True)

        # Check if all proofs are completed
        all_proofs_computed = all(sub_model.is_completed for sub_model in onnx_model.sub_models)
        if all_proofs_computed:
            logger.info('All proofs computed successfully.')
        else:
            logger.warning('Some proofs failed to compute.')

    # def compute_proof(self, onnx_model: OnnxModel):
    #     """
    #     Computes zero-knowledge proofs for the given ONNX model by distributing
    #     sub-models to available workers.
    #     """

    #     logger.info(f'Starting proof computation for every sub model..')

    #     def send_proof_request(worker: Worker, sub_model: OnnxModel):
    #         """
    #         Sends a proof computation request to a specific worker.
    #         """
    #         try:
    #             with grpc.insecure_channel(worker.address) as channel:
    #                 stub = pb2_grpc.ZKPServiceStub(channel)
    #                 # Prepare the request message
    #                 request = pb2.ProofRequest(
    #                     model_path=sub_model.onnx_model_path,
    #                     input_data=sub_model.input_data
    #                 )
    #                 # Send request and get response
    #                 response = stub.ComputeProof(request)
    #                 # Process response
    #                 if response.success:
    #                     sub_model.computed_proof = response.proof
    #                     sub_model.is_completed = True
    #                     logger.info(f'Proof computed for sub-model {sub_model.id} by worker {worker.address}')
    #                 else:
    #                     logger.error(f'Proof computation failed for sub-model {sub_model.id} by worker {worker.address}')
    #         except Exception as e:
    #             logger.error(f'Exception occurred while computing proof for sub-model {sub_model.id} on worker {worker.address}: {e}')
    #         finally:
    #             worker.is_free = True  # Mark worker as free once done

    #     # Initialize the task queue with sub-models
    #     task_queue = Queue()
    #     for sub_model in onnx_model.sub_models:
    #         task_queue.put(sub_model)

    #     # Define a function to process tasks
    #     def process_tasks():
    #         while not task_queue.empty():
    #             # Find a free worker
    #             free_worker = None
    #             while free_worker is None:
    #                 for worker in self.workers:
    #                     if worker.is_free:
    #                         free_worker = worker
    #                         break
    #                 if free_worker is None:
    #                     time.sleep(10)  # Wait for some time before checking again
                
    #             # Get the next sub-model from the queue
    #             sub_model = task_queue.get()
    #             free_worker.is_free = False
    #             # Submit the proof request task
    #             executor.submit(send_proof_request, free_worker, sub_model)
        
    #     # Create a ThreadPoolExecutor for handling tasks
    #     with ThreadPoolExecutor(max_workers=self.number_of_workers) as executor:
    #         # Start processing tasks
    #         process_tasks()

    #     # Wait for all tasks to complete
    #     executor.shutdown(wait=True)

    #     # Check if all proofs are completed
    #     all_proofs_computed = all(sub_model.is_completed for sub_model in onnx_model.sub_models)
    #     if all_proofs_computed:
    #         logger.info('All proofs computed successfully.')
    #     else:
    #         logger.warning('Some proofs failed to compute.')

        
    def check_worker_connections(self, worker_addresses):
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
                        logger.error(f"Cannot connect to worker at {worker_address}") 
            except:
                logger.error(f"Cannot connect to worker at {worker_address}")
    

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    
    if not os.path.exists(config.model.onnx_file):
            raise FileNotFoundError(f"The specified file '{config.model.onnx_file}' does not exist.")
    
    if not os.path.exists(config.model.input_file):
            raise FileNotFoundError(f"The specified file '{config.model.input_file}' does not exist.")  
    
    prover = ZKPProver(config=config)

    logger.info(f'Started Processing: {config.model}')
    # logging.info(f'Model Info: {onnx_model_for_proving.info}')
    prover.prepare_model_for_distrubuted_proving(config.model.onnx_file, config.model.input_file)



if __name__ == '__main__':
    main()