
import hydra
from omegaconf import DictConfig
import logging
from typing import List
import os
import json
from onnx import ModelProto
import csv
import datetime
from enum import Enum
import onnx
import ezkl
import onnx
import onnxruntime as ort
import json
import numpy as np
import os
from onnx.utils import Extractor
from collections import OrderedDict
import copy
from onnx import shape_inference

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger("ZKPProver")

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

        #check if the model and input data are valid
        result = self.run_model_inference()


    def run_model_inference(self):
        input_data = fo(self.input_data_path)
        if 'input_data' in input_data:
            input_tensor = input_data['input_data']      
        session = ort.InferenceSession(self.onnx_model_path)
        # Ensure input matches ONNX model requirements
        input_name = session.get_inputs()[0].name
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        return outputs


@hydra.main(version_base=None, config_path="conf", config_name="config")
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