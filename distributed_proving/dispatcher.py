import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import logging
from usefulutils import  analyze_onnx_model_for_zk_proving, load_onnx_model, read_json_file_to_dict, count_onnx_model_operations
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
import datetime
import hashlib
# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")

class Worker():   
    def __init__(self, id: int, address):
        self.id = id
        self.address = address
        self.is_free = True
        self.channel:Channel = None

class OnnxModel():   
    def __init__(self, 
                 id:str, 
                 input_data:str, 
                 data_folder:str,
                 onnx_model_path:str, 
                 combined_node_indices = []):
        if onnx_model_path is None:
            raise TypeError("Model path or model proto must be provided")

        self.onnx_model_path = onnx_model_path
        self.data_folder = data_folder
        self.id = id
        self.isprocessed = False
        self.computed_witness = None
        self.computed_proof = None
        self.input_data = input_data
        self.model_info = {'model_id': self.id,'combined_splits': combined_node_indices}
        self.model_info.update(analyze_onnx_model_for_zk_proving(self.onnx_model_path))

    def get_model_prorto(self):
        if self.onnx_model_proto is None:
            self.onnx_model_proto = load_onnx_model(self.onnx_model_path)
        return self.onnx_model_proto
    
class ZKPProvingDispatcherService():
    def __init__(self, config: DictConfig):
        self.workers:List[Worker] = []
        self.model_name:str = config.model.name
        self.model_onnx_file:str = config.model.onnx_file
        self.model_input_file:str = config.model.input_file
        self.split_group_size = config.model.split_group_size
        self.group_split_lists = config.model.group_splits
        self.cache_setup_files = config.cache_setup_files
        self.spot_test = config.spot_test
        # Get the current timestamp in a desired format (e.g., 'YYYY-MM-DD_HH-MM-SS')
        datetimenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_folder = os.path.join('logs', self.model_name, datetimenow)
        # os.makedirs(self.log_folder, exist_ok=True)
        self.report_file = os.path.join(self.log_folder, 'performance_logs.csv')
        self.models_info_file = os.path.join(self.log_folder, 'models_info.csv')


    def validate_worker_connections(self, worker_addresses):
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
            except Exception as e:
                logger.error(f"Cannot connect to worker at {worker_address}, error: {str(e)}")
    
    
    def prepare_for_proof_generation(self): 
        models_to_prove = []

        logger.info(f'Analyzing model {self.model_onnx_file}..' )
        if self.split_group_size is None:
            logger.info(f'No split size provided. Proving the model as a whole..')
            
            global_model = OnnxModel(
                id=self.model_name, 
                input_data=read_json_file_to_dict(self.model_input_file), 
                onnx_model_path= self.model_onnx_file,
                data_folder=self.log_folder) 
            models_to_prove.append(global_model)
            
            file_exists = os.path.isfile(self.models_info_file)
            with open(self.models_info_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=global_model.models_info_file.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(global_model.model_info)
        
        else:
            logger.info(f'Collecting intermediate inference outputs..')
            intermediate_inference_outputs = get_intermediate_outputs(self.model_onnx_file, self.model_input_file)
            logger.info(f'Intermediate inference outputs collected')

            if self.group_split_lists is None:    
                logger.info(f'Splitting model based on the configured group size {self.split_group_size}')
            else:
                logger.info(f'Splitting model based on the configured group size {self.split_group_size} and group split lists: {self.group_split_lists} ..')

            all_sub_models = split_onnx_model_at_every_node(self.model_onnx_file, self.model_input_file, intermediate_inference_outputs,'tmp',False)
            
            if self.split_group_size < len(all_sub_models):
                sub_models = self.group_models(all_sub_models, self.split_group_size, self.group_split_lists, False)
            else:
                sub_models = all_sub_models
            
            logger.info(f'Total number of sub-models for distributed proving: {len(sub_models)}' )

            logger.info(f'Preparing sub-models..')
            for idx, group in enumerate(sub_models):
                merged_model, combined_node_indices = merge_onnx_models(group)
                inputs = self.get_model_inputs(merged_model, intermediate_inference_outputs) 
                flattened_inputs =  []
                for line in inputs:
                    flattened_inputs.append(line.flatten().tolist())
                input_data = {"input_data": flattened_inputs}

                sub_model = OnnxModel(id=f'{self.model_name}({len(sub_models)}_splits)_sub_model_{idx+1}',
                                      input_data=input_data,
                                      model_proto= merged_model, 
                                      combined_node_indices = combined_node_indices)
                models_to_prove.append(sub_model)

                if save_ezkl_settings:
                    file_exists = os.path.isfile(self.ezkl_settings_file)
                    with open(self.ezkl_settings_file, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=sub_model.ezkl_settings.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(sub_model.ezkl_settings)

        return models_to_prove
    
    def get_free_worker(self):
        while True:
            for worker in self.workers:
                if worker.is_free:
                    return worker
            time.sleep(10)
    
    def compute_proof_for_model(self, model: OnnxModel, worker: Worker):
        performance_metrics = None  # Initialize to avoid undefined reference
        try:
            channel = grpc.insecure_channel(worker.address)
            stub = pb2_grpc.ZKPWorkerServiceStub(channel)
            request = pb2.ProofRequest(
                    model_id=model.id,
                    onnx_model=model.model_proto.SerializeToString(),
                    input_data=json.dumps(model.input_data),
                    cache_setup_files=False)
            response = stub.ComputeProof(request)
            request_id = response.request_id
            logger.info(f'Started proof computation for sub-model {model.id} on worker {worker.address}. Request ID: {request_id}')

            #poll every few seconds to check if the proof computation is completed
            polling_exception_count = 0
            time.sleep(10)  # Optional: Add a short delay before retrying
            while True:
                try:
                    job_status = stub.CheckProofStatus(pb2.ProofStatusRequest(request_id=request_id), timeout=30)
                    if job_status.is_completed:
                        model.isprocessed = True
                        if job_status.is_success:
                            model.computed_proof = job_status.proof
                            logger.info(f'Successfully generated proof for sub-model {model.id} on worker {worker.address}. Request ID: {request_id}')
                        else:
                            logger.error(f'Proof computation failed for sub-model {model.id} on worker {worker.address}. Request ID: {request_id}') 
                        performance_metrics = json.loads(job_status.performance_data)
                        break
                    else:
                        logger.info(f'Proof computation in progress for sub-model {model.id} on worker {worker.address}. Waiting for 10 seconds before retrying.')
                        time.sleep(10)
                except Exception as e:
                    polling_exception_count += 1
                    # if polling_exception_count >= 3:
                    #     logger.error(f'Polling for proof computation status for sub-model {model.id} on worker {worker.address} failed after 5 retries. Exiting..')
                    #     break
                    time.sleep(15)
                    channel.close() # Close the old channel
                    channel = grpc.insecure_channel(worker.address) # Reconnect to the worker
                    stub = pb2_grpc.ZKPWorkerServiceStub(channel)
                    continue
            self.write_report(worker.address, model.model_info, job_status.is_success, performance_metrics)
        except Exception as e:
            logger.error(f'Proof computation for sub-model {model.id} on worker {worker.address} failed with error: {e}. Aborting...')
            worker.is_free = True
        finally:
            channel.close()
            worker.is_free = True


    def write_report(self, worker_address, model_info: dict, is_successful:bool, performance_data: dict):

        if is_successful:
            if 'fft_data' in performance_data:
                fft_folder = os.path.join(self.log_folder, 'ffts')
                if not os.path.exists(fft_folder):
                    os.makedirs(fft_folder)
            fft_data = json.loads(performance_data.pop('fft_data'))
            fft_file = os.path.join(fft_folder, f'{model_info["model_id"]}_ffts.csv')
            with open(fft_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fft_data[0].keys())
                writer.writeheader()
                writer.writerows(fft_data)

            if 'msm_data' in performance_data:
                msm_folder = os.path.join(self.log_folder, 'msms')
                if not os.path.exists(msm_folder):
                    os.makedirs(msm_folder)
                msm_data = json.loads(performance_data.pop('msm_data'))
                msm_file = os.path.join(msm_folder, f'{model_info["model_id"]}_msms.csv')
                with open(msm_file, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=msm_data[0].keys())
                    writer.writeheader()
                    writer.writerows(msm_data)
            
            report_data = {**model_info, **performance_data}
            report_data['worker_address'] = worker_address
            success_report_file = os.path.join(self.log_folder, 'proof_performance.csv')
            file_exists = os.path.isfile(success_report_file)
            with open(success_report_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=report_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(report_data)
        else:
            report_data = {**model_info, **performance_data}
            report_data['worker_address'] = worker_address
            failure_report_file = os.path.join(self.log_folder, 'failed_proofs.csv')
            file_exists = os.path.isfile(failure_report_file)
            with open(failure_report_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=report_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(report_data)

    def group_models(self, models:OrderedDict, n, group_split_lists = None, spot_test_only = False):
        grouped_splits = []
        if group_split_lists is not None:
            for target_list in group_split_lists:
                tmp = {}
                for split_id in target_list:
                    item_key = f'split_model_{split_id}'
                    if item_key in models:
                        tmp[item_key] = models[item_key]
                        # tmp.append((item_key, models.pop(item_key)))
                grouped_splits.append(tmp)
            if spot_test_only:
                return grouped_splits
        
        for tmp in grouped_splits:
            for key in tmp.keys():
                if key in models:
                    models.pop(key)
           
        items = list(models.items())
        temp2 = [dict(items[i:i+n]) for i in range(0, len(items), n)]
        grouped_splits.extend(temp2)
        return grouped_splits
    
    def get_model_inputs(self, model, intermediate_values):
        input_data = []
        for input_tensor in model.graph.input:
            input_data.append(intermediate_values[input_tensor.name])
        return input_data
            
        

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):

    if not os.path.exists(config.model.onnx_file):
            raise FileNotFoundError(f"The specified file '{config.model.onnx_file}' does not exist.")
    
    if not os.path.exists(config.model.input_file):
            raise FileNotFoundError(f"The specified file '{config.model.input_file}' does not exist.")  

    dispatcher = ZKPProvingDispatcher(config=config)

    #check worker connections
    dispatcher.validate_worker_connections(config.worker_addresses)
    models_to_prove:List[OnnxModel] = dispatcher.prepare_for_proof_generation()

    if not config.generate_ezkl_settings_only:

        if len(models_to_prove) == 1:
            logger.info(f'Starting proof computation for the global model..')
        else:
            logger.info(f'Starting proof computation for {len(models_to_prove)} sub-models..')
        
        with ThreadPoolExecutor(max_workers=len(prover.workers)) as executor:
            future_to_model = {}
            # Submit tasks for each model to compute the proof
            for model in models_to_prove:
                worker = prover.get_free_worker()
                worker.is_free = False
                future = executor.submit(prover.compute_proof_for_model, model, worker)
                future_to_model[future] = model

        all_proofs_computed = all(sub_model.computed_proof for sub_model in models_to_prove)

    if config.generate_ezkl_settings_only:
        logger.info('EZKL settings generated successfully for all models.')
    else:
        if all_proofs_computed:
            logger.info('All proofs computed successfully.')
        else:
            logger.warning('Some proofs failed to compute.')


if __name__ == '__main__':
    main()