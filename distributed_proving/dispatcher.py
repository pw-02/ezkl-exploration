import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import logging
from utils import  analyze_onnx_model_for_zk_proving, load_onnx_model, read_json_file_to_dict
from split_model import get_intermediate_outputs, split_onnx_model_at_every_node,  merge_onnx_models
from typing import List
import os
from queue import Queue
from grpc import Channel 
import json
from onnx import ModelProto
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from collections import OrderedDict

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")

class Worker():   
    def __init__(self, id: int, address):
        self.id = id
        self.address = address
        self.is_free = True
        self.channel:Channel = None

class OnnxModel ():   
    def __init__(self, id: int, input_data:str, onnx_model_path:str = None, model_proto:ModelProto = None):

        if onnx_model_path is None and model_proto is None:
            raise TypeError("Model path or model proto must be provided")
        elif model_proto:
            self.model_proto = model_proto
        else:
            self.model_proto = load_onnx_model(onnx_model_path)
        self.id = id
        self.computed_witness = None
        self.computed_proof = None
        self.is_completed = False
        self.input_data = input_data
        self.sub_models: List[OnnxModel] = []
        self.info = {'model_id': self.id}
        self.info.update(analyze_onnx_model_for_zk_proving(onnx_model=self.model_proto))
        # self.info (analyze_onnx_model_for_zk_proving(onnx_model=self.model_proto))


class ZKPProver():
    def __init__(self, config: DictConfig):
        self.workers:List[Worker] = []
        self.check_worker_connections(config.worker_addresses)
        self.report_path = config.report_path
    
    def write_report(self, worker_address, model_info: dict ,performance_data: dict):

        report_data = {**model_info, **performance_data}
        report_data['worker_address'] = worker_address

        file_exists = os.path.isfile(self.report_path)

        with open(self.report_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=report_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(report_data)


    def group_models(self, models:OrderedDict, n):
        items = list(models.items())  # Convert dictionary items to a list of tuples
        grouped_items = [dict(items[i:i+n]) for i in range(0, len(items), n)]
        return grouped_items
    
    def get_model_inputs(self, model, intermediate_values):
        input_data = []
        for input_tensor in model.graph.input:
            input_data.append(intermediate_values[input_tensor.name])
        return input_data
    
    def prepare_model_for_distributed_proving(self, model_name:str, onnx_model_path:str, json_input_file:str, split_group_size = None):
        logger.info(f'Analyzing model...')

        #get the output tensor(s) of every node node in the model during inference
        global_model = OnnxModel(id = f'global_{model_name}', 
                                 input_data=read_json_file_to_dict(json_input_file), 
                                 onnx_model_path= onnx_model_path)
        
        logger.info(f'Num model params: {global_model.info["num_model_params"]}, Num rows in zk circuit: {global_model.info["zk_circuit_num_rows"]}, Number of nodes: {global_model.info["num_model_ops"]}')
        
        if split_group_size is None:
            logger.info(f'No split size provided. Proving the model as a whole..')
            global_model.sub_models.append(global_model)
        else:
            logger.info(f'Splitting model based on the configured group size..')
            node_inference_outputs = get_intermediate_outputs(onnx_model_path, json_input_file)
            all_sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input_file, node_inference_outputs)

            if split_group_size < len(all_sub_models):
                grouped_models = self.group_models(all_sub_models, split_group_size)
            else:
                grouped_models = all_sub_models

            logger.info(f'Total number of sub-models for distributed proving: {len(grouped_models)}' )

            for idx, group in enumerate(grouped_models):
                logger.info(f'Preparing sub-model {idx+1}..')
                
                merged_model = merge_onnx_models(group)
                inputs = self.get_model_inputs(merged_model, node_inference_outputs)

                #flatttern input data so it can be sent to each worker as json
                flattened_inputs =  []
                for line in inputs:
                    flattened_inputs.append(line.flatten().tolist())
                input_data = {"input_data": flattened_inputs}
                sub_model = OnnxModel(id=f'{model_name}_sub_model_{idx+1}/{len(grouped_models)}',
                                      input_data=input_data,
                                      model_proto= merged_model)
                global_model.sub_models.append(sub_model)
        
        self.compute_proof(global_model)

    def compute_proof(self, onnx_model: OnnxModel):
        logger.info(f'Starting proof computation for sub-models..')
        
        def send_proof_request(worker: Worker, sub_model: OnnxModel):
            try:
                channel = grpc.insecure_channel(worker.address)
                stub = pb2_grpc.ZKPWorkerServiceStub(channel)
                request = pb2.ProofRequest(
                    model_id=sub_model.id,
                    onnx_model=sub_model.model_proto.SerializeToString(),
                    input_data=json.dumps(sub_model.input_data)
                )
                response = stub.ComputeProof(request)
                request_id = response.request_id

                if request_id:
                    logger.info(f'Started proof computation for sub-model {sub_model.id} on worker {worker.address}. Request ID: {request_id}')
                    time.sleep(10)  # Optional: Add a short delay before retrying
                    polling_exccpetion_count = 0
                    while True:
                        try:
                            status_request = pb2.ProofStatusRequest(request_id=request_id)
                            status_response = stub.CheckProofStatus(status_request, timeout=30)

                            if status_response.success:
                                    sub_model.computed_proof = status_response.proof
                                    sub_model.is_completed = True
                                    logger.info(f'Proof computation completed for sub-model {sub_model.id} by worker {worker.address}')
                                    performance_data = json.loads(status_response.performance_data)
                                    self.write_report(worker.address, sub_model.info, performance_data)
                                    break
                            else:
                                    logger.info(f'Proof computation in progress for sub-model {sub_model.id} on worker {worker.address}. Waiting for 10 seconds before retrying.')
                                    time.sleep(10)
                                                
                        except Exception as e:
                            polling_exccpetion_count += 1
                            # if polling_exccpetion_count > 25:
                            #     logger.error(f'Proof computation failed for sub-model {sub_model.id} by worker {worker.address}. Aborting...')
                            #     break
                            # else:
                                # logger.error(f'RPC exception occurred while polling for proof status for sub-model {sub_model.id}: {e}. Retrying...')
                            time.sleep(30)  # Optional: Add a short delay before retrying
                            channel.close() # Close the old channel
                            channel = grpc.insecure_channel(worker.address) # Reconnect to the worker
                            stub = pb2_grpc.ZKPWorkerServiceStub(channel) # Reinitialize the stub
                            continue
            except Exception as e:
                logger.error(f'Proof computation failed for sub-model {sub_model.id} by worker {worker.address}. Aborting...')
            finally:
                channel.close() # Close the channel
                worker.is_free = True

        # Initialize the task queue with sub-models
        task_queue = Queue()
        if len(onnx_model.sub_models) >0:
            for sub_model in onnx_model.sub_models:
                task_queue.put(sub_model)
        else:
            task_queue.put(onnx_model)
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
                    stub = pb2_grpc.ZKPWorkerServiceStub(channel)
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
    prover.prepare_model_for_distributed_proving(config.model.name, config.model.onnx_file, config.model.input_file, config.model.split_group_size)


if __name__ == '__main__':
    main()


    
    # def prepare_model_for_distributed_proving(self, model_name:str, onnx_model_path:str, json_input_file:str, num_splits = 1):
    #     logger.info(f'Analyzing model...')

    #     #get the output tensor(s) of every node node in the model during inference
    #     global_model = OnnxModel(id = f'global_{model_name}', 
    #                              input_data=read_json_file_to_dict(json_input_file), 
    #                              onnx_model_path= onnx_model_path)
        
    #     logger.info(f'Num model params: {global_model.info["num_model_params"]}, Num rows in zk circuit: {global_model.info["zk_circuit_num_rows"]}, Number of nodes: {global_model.info["num_model_ops"]}')
        
    #     if num_splits >1:

    #         logger.info(f'Splitting model for distrubuted proving..')
    #         node_inference_outputs = get_intermediate_outputs(onnx_model_path, json_input_file)
    #         all_sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input_file, node_inference_outputs)
            
    #         grouped_sub_models = self.group_models(all_sub_models, len(all_sub_models)//num_splits)
            
    #         # sub_models = split_onnx_model(onnx_model_path, json_input_file, node_outputs, 50, 'tmp')

    #         #add in some logic here later if we need to combine split models for load balancing

    #         for idx, (sub_model_poto, input_data) in enumerate(sub_models):
    #             if idx+1 == 3 or idx+1 == 95:
    #                 sub_model = OnnxModel(
    #                     id=f'{model_name}_part_{idx+1}',
    #                                     input_data=input_data,
    #                                     model_proto= sub_model_poto)
    #                 global_model.sub_models.append(sub_model)
        
    #     self.compute_proof(global_model)