import grpc
import time
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import logging
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
from enum import Enum
import onnx
import ezkl
import onnx
import onnxruntime as ort
from onnx import shape_inference, ModelProto
import json
import numpy as np
import os
from onnx.utils import Extractor
from collections import OrderedDict
import copy
from onnx import shape_inference



# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPProver")



def format_model_input(input_path, input_shape, input_type, idx = 0):
    # input_shape = [-1 if dim == 'batch_size' else dim for dim in input_shape]
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    if input_type == 'tensor(float)':

        input_data = np.array(input_data['input_data'][idx], dtype=np.float32)
        if len(input_shape)>1:
            try:
                input_data = input_data.reshape(input_shape)
                return input_data
            except ValueError as e:
                raise ValueError(f"Input data cannot be reshaped from shape {input_data.shape} to the expected shape {input_shape}: {e}")

    elif input_type == 'tensor(int64)': 
        if len(input_shape)>0:
            input_data = np.array(input_data['input_data'][idx], dtype=np.int64)
        else:
            input_data = np.array(input_data['input_data'][0][0],dtype=np.int64) 

    elif input_type == 'tensor(int32)':
        return input_data
    
    
    

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    
class OnnxModelToProve():
    def __init__(self, job_id, job_name, input_data, onnx_model_path, data_directory):
        self.job_id = job_id
        self.model_name = job_name
        self.input_data = input_data
        self.onnx_model_path = onnx_model_path
        self.status = JobStatus.PENDING
        self.data_directory = data_directory
    
    def generate_model_info(self):
        tmp_settings_file = 'tmp_settings.json'
        onnx_model = onnx.load(self.onnx_model_path)
        num_model_ops = len(onnx_model.graph.node)
        num_model_params =  0
        for initializer in onnx_model.graph.initializer:
            param_array = onnx.numpy_helper.to_array(initializer)
            num_model_params += param_array.size
        #get ezkl settings
        ezkl.gen_settings(self.onnx_model_path, tmp_settings_file)
        try:
            with open(tmp_settings_file, 'r') as f:
                ezkl_settings = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading JSON settings file: {e}")
            ezkl_settings = {}

        self.model_info = {"num_model_ops": num_model_ops,"num_model_params": num_model_params,}
        self.model_info = {**self.model_info, **ezkl_settings}
        #delete temp file
        os.remove(tmp_settings_file)  



class GlobalProvingJob():
    def __init__(self, job_name, input_data_path, onnx_model_path, num_of_splits):
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.num_of_splits = num_of_splits
        self.status = JobStatus.PENDING
        self.model_to_prove: List[OnnxModelToProve] = []

    def pepare_for_processing(self, save_ezkl_settings = False):
        #verify the input data path and model path 
        if not os.path.exists(self.input_data_path):
            logger.error(f'Input data path does not exist: {self.input_data_path}')
            raise FileNotFoundError(f"Input data path does not exist: {self.input_data_path}")
        if not os.path.exists(self.onnx_model_path):
            logger.error(f'ONNX model path does not exist: {self.onnx_model_path}')
            raise FileNotFoundError(f"ONNX model path does not exist: {self.onnx_model_path}")

        result = self.run_model_inference()

    def run_model_inference(self):
        session = ort.InferenceSession(self.onnx_model_path)
    
        # Ensure input matches ONNX model requirements
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_type = session.get_inputs()[0].type   

        input_tensor = format_model_input(self.input_data_path,
                                       input_shape,
                                       input_type)
 
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        return outputs
    


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):

    if not os.path.exists(config.model.onnx_file):
            raise FileNotFoundError(f"The specified file '{config.model.onnx_file}' does not exist.")
    
    if not os.path.exists(config.model.input_file):
            raise FileNotFoundError(f"The specified file '{config.model.input_file}' does not exist.")  
    
    job = GlobalProvingJob(
                             job_name=config.model.name,
                             input_data_path=config.model.input_file,
                             onnx_model_path=config.model.onnx_file,
                             num_of_splits=config.model.split_group_size)
    
    job.pepare_for_processing(save_ezkl_settings=True)
    

if __name__ == '__main__':
    main()