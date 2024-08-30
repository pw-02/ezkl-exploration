import grpc
import asyncio
import json
import logging
import os
from typing import List, Dict
from collections import OrderedDict
from grpc.aio import Channel
from onnx import ModelProto
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
from utils import analyze_onnx_model_for_zk_proving, load_onnx_model, read_json_file_to_dict, count_onnx_model_operations
from split_model import get_intermediate_outputs, split_onnx_model_at_every_node, merge_onnx_models
import csv
import hydra
from omegaconf import DictConfig

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")

class Worker:
    def __init__(self, id: int, address):
        self.id = id
        self.address = address
        self.is_free = True
        self.channel: Channel = None
        self.stub = None

class OnnxModel:
    def __init__(self, id: int, input_data: str, onnx_model_path: str = None, model_proto: ModelProto = None, combined_node_indices = None):
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
        self.info['combined_splits'] = combined_node_indices

class ZKPProver:
    def __init__(self, config: DictConfig):
        self.workers: List[Worker] = []
        self.check_worker_connections(config.worker_addresses)
        self.report_path = config.report_path

    def write_report(self, worker_address, model_info: dict, performance_data: dict):
        report_data = {**model_info, **performance_data}
        report_data['worker_address'] = worker_address

        file_exists = os.path.isfile(self.report_path)

        with open(self.report_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=report_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(report_data)

    def group_models(self, models: OrderedDict, n, group_split_lists=None, spot_test_only=False):
        grouped_splits = []
        if group_split_lists is not None:
            for target_list in group_split_lists:
                tmp = {}
                for split_id in target_list:
                    item_key = f'split_model_{split_id}'
                    if item_key in models:
                        tmp[item_key] = models[item_key]
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

    async def prepare_model_for_distributed_proving(self, model_name: str,
                                                    onnx_model_path: str,
                                                    json_input_file: str,
                                                    split_group_size=None,
                                                    cache_setup_files=False,
                                                    group_split_lists=None,
                                                    spot_test=False):
        logger.info(f'Analyzing model...')
        node_inference_outputs = get_intermediate_outputs(onnx_model_path, json_input_file)
        all_sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input_file, node_inference_outputs)

        total_sub_models = len(all_sub_models)

        global_model = OnnxModel(id=f'global_{model_name}',
                                 input_data=read_json_file_to_dict(json_input_file),
                                 onnx_model_path=onnx_model_path,
                                 combined_node_indices=list(range(1, total_sub_models + 1)))

        logger.info(f'Num model params: {global_model.info["num_model_params"]}, Num rows in zk circuit: {global_model.info["zk_circuit_num_rows"]}, Number of nodes: {global_model.info["num_model_ops"]}')

        if split_group_size is None:
            logger.info(f'No split size provided. Proving the model as a whole..')
            global_model.sub_models.append(global_model)
        else:
            if group_split_lists is None:
                logger.info(f'Splitting model based on the configured group size {split_group_size}')
            else:
                logger.info(f'Splitting model based on the configured group size {split_group_size} and group split lists: {group_split_lists} ..')

            if split_group_size < len(all_sub_models):
                grouped_models = self.group_models(all_sub_models, split_group_size, group_split_lists, spot_test)
            else:
                grouped_models = all_sub_models

            logger.info(f'Total number of sub-models for distributed proving: {len(grouped_models)}')

            for idx, group in enumerate(grouped_models):
                logger.info(f'Preparing sub-model {idx+1}..')

                merged_model, combined_node_indices = merge_onnx_models(group)
                inputs = self.get_model_inputs(merged_model, node_inference_outputs)

                flattened_inputs = []
                for line in inputs:
                    flattened_inputs.append(line.flatten().tolist())
                input_data = {"input_data": flattened_inputs}

                sub_model = OnnxModel(id=f'{model_name}_sub_model_{idx+1}/{len(grouped_models)}',
                                      input_data=input_data,
                                      model_proto=merged_model, combined_node_indices=combined_node_indices)
                global_model.sub_models.append(sub_model)

        await self.compute_proof(global_model, cache_setup_files)

    async def send_proof_request(self, worker: Worker, sub_model: OnnxModel):
        try:
            channel = grpc.aio.insecure_channel(worker.address)
            stub = pb2_grpc.ZKPWorkerServiceStub(channel)
            request = pb2.ProofRequest(
                model_id=sub_model.id,
                onnx_model=sub_model.model_proto.SerializeToString(),
                input_data=json.dumps(sub_model.input_data),
                cache_setup_files=False
            )
            response = await stub.ComputeProof(request)
            request_id = response.request_id

            if request_id:
                logger.info(f'Started proof computation for sub-model {sub_model.id} on worker {worker.address}. Request ID: {request_id}')
                while True:
                    try:
                        status_request = pb2.ProofStatusRequest(request_id=request_id)
                        status_response = await stub.CheckProofStatus(status_request, timeout=30)

                        if status_response.success:
                            sub_model.computed_proof = status_response.proof
                            sub_model.is_completed = True
                            logger.info(f'Proof computation completed for sub-model {sub_model.id} by worker {worker.address}')
                            performance_data = json.loads(status_response.performance_data)
                            self.write_report(worker.address, sub_model.info, performance_data)
                            break
                        else:
                            logger.info(f'Proof computation in progress for sub-model {sub_model.id} on worker {worker.address}. Waiting for 10 seconds before retrying.')
                            await asyncio.sleep(10)

                    except Exception as e:
                        logger.error(f'RPC exception occurred while polling for proof status for sub-model {sub_model.id}: {e}. Retrying...')
                        await asyncio.sleep(30)  # Optional: Add a short delay before retrying
                        continue
        except Exception as e:
            logger.error(f'Proof computation failed for sub-model {sub_model.id} by worker {worker.address}. Aborting...')
        finally:
            await channel.close()
            worker.is_free = True

    async def compute_proof(self, onnx_model: OnnxModel, cache_setup_files):
        logger.info(f'Starting proof computation for sub-models..')

        task_queue = asyncio.Queue()
        if len(onnx_model.sub_models) > 0:
            for sub_model in onnx_model.sub_models:
                await task_queue.put(sub_model)
        else:
            await task_queue.put(onnx_model)

        async def process_tasks():
            while not task_queue.empty():
                free_worker = None
                while free_worker is None:
                    for worker in self.workers:
                        if worker.is_free:
                            free_worker = worker
                            break
                    if free_worker is None:
                        await asyncio.sleep(10)

                sub_model = await task_queue.get()
                free_worker.is_free = False
                asyncio.create_task(self.send_proof_request(free_worker, sub_model))

        await process_tasks()

    async def check_worker_connections(self, worker_addresses: List[str]):
        self.workers = [Worker(id=i, address=addr) for i, addr in enumerate(worker_addresses)]

        async def ping_worker(worker: Worker):
            try:
                # Create an asynchronous channel to the worker
                channel = grpc.aio.insecure_channel(worker.address)
                stub = pb2_grpc.ZKPWorkerServiceStub(channel)

                # Optionally, you might want to implement a ping RPC method in your service
                # Here we assume there's a HealthCheck method for simplicity
                health_check_request = pb2.HealthCheckRequest()
                response = await stub.HealthCheck(health_check_request)
                
                if response.status == pb2.HealthCheckResponse.STATUS_OK:
                    logger.info(f'Worker {worker.address} is reachable and responsive.')
                    worker.is_free = True
                else:
                    logger.warning(f'Worker {worker.address} is not responsive. Status: {response.status}')
                    worker.is_free = False

            except Exception as e:
                logger.error(f'Failed to connect to worker {worker.address}. Error: {e}')
                worker.is_free = False
            finally:
                await channel.close()

        # Create tasks to ping each worker
        ping_tasks = [ping_worker(worker) for worker in self.workers]
        await asyncio.gather(*ping_tasks)

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
async def main(config: DictConfig):
    if not os.path.exists(config.model.onnx_file):
        raise FileNotFoundError(f"The specified file '{config.model.onnx_file}' does not exist.")
    
    if not os.path.exists(config.model.input_file):
        raise FileNotFoundError(f"The specified file '{config.model.input_file}' does not exist.")  
    
    prover = ZKPProver(config=config)

    logger.info(f'Started Processing: {config.model}')

    # Check worker connections asynchronously
    await prover.check_worker_connections(config.worker_addresses)

    # Prepare the model for distributed proving
    await prover.prepare_model_for_distributed_proving(
        model_name=config.model.name, 
        onnx_model_path=config.model.onnx_file,
        json_input_file=config.model.input_file, 
        split_group_size=config.model.split_group_size, 
        cache_setup_files=config.cache_setup_files,
        group_split_lists=config.group_splits,
        spot_test=config.spot_test
    )

if __name__ == '__main__':
    asyncio.run(main())