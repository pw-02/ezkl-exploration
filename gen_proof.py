# Standard library
import os
import json
import csv
import time
import logging
import threading
import multiprocessing
from enum import Enum
from datetime import datetime, timezone
from functools import wraps
from collections import OrderedDict
from typing import List
# Third-party libraries
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from onnx.utils import Extractor
import ezkl
import hydra
from omegaconf import DictConfig
import shutil  # at the top of your file
# Local modules
from utils.resource_monitor import log_system_usage


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

# Decorator to time functions
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        return execution_time
    return wrapper


logger = logging.getLogger("dizt-zkml")
logger.setLevel(logging.INFO)
# Remove any existing file handlers
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
# Ensure logging only prints to console
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, force=True)


def file_exists(file_path):
    return os.path.exists(file_path)

def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)  # Expecting a nested list from np.array().tolist()

def format_model_input(input_data_path, expected_shape, input_type, idx = 0):
    expected_shape = [-1 if dim == 'batch_size' else dim for dim in expected_shape]
    input_data = load_json_input(input_data_path)['input_data']
    
    if input_type == 'tensor(float)':
        input_data = np.array(input_data, dtype=np.float32)  # Convert back to NumPy array
        reshaped_input = input_data.reshape(expected_shape)
    
    elif input_type == 'tensor(int64)':
        if len(expected_shape)>0:
            reshaped_input = np.array(input_data[idx], dtype=np.int64)
        else:
            reshaped_input = np.array(input_data[0][0],dtype=np.int64) 
    
    if 'gpt' in str(input_data_path).lower():
        reshaped_input = np.reshape(input_data, (1, 64))  # Shape: (1, 64)

    #import matplotlib.pyplot as plt
    # plt.imshow(reshaped_input.squeeze(), cmap='gray')  # Removes batch and channel dimensions
    # plt.title("Input Image")
    # plt.axis("off")
    # plt.show()
    return reshaped_input

def read_csv_into_dict(file_path):
    data = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                data[key] = value
    return data


def get_fft_summary(fft_file, prefix):
        fft_metrics = {}
        df = pd.read_csv(fft_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of FFTs
        fft_metrics[f'{prefix}_fft_count'] = int(len(df))
        fft_metrics[f'{prefix}_fft_largest'] = int(df['size'].max())
        fft_metrics[f'{prefix}_fft_total_time(s)'] = float(df['duration(s)'].sum())
        fft_metrics[f'{prefix}_fft_avg_time(s)'] = float(df['duration(s)'].mean())
        fft_metrics[f'{prefix}_fft_device'] = str(df['device'].iloc[0])
        # Convert the DataFrame to a dictionary
        # data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        # fft_data = json.dumps(data_dict)
        # fft_metrics[f'{prefix}_fft_data'] = fft_data
        return fft_metrics

def get_msm_summary(msm_file, prefix):
        msm_metrics = {}
        df = pd.read_csv(msm_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of MSMs
        msm_metrics[f'{prefix}_msm_count'] = int(len(df))
        msm_metrics[f'{prefix}_msm_largest'] = int(df['num_coeffs'].max())
        msm_metrics[f'{prefix}_msm_total_time(s)'] = float(df['duration(s)'].sum())
        msm_metrics[f'{prefix}_msm_avg_time(s)'] = float(df['duration(s)'].mean())
        msm_metrics[f'{prefix}_msm_device'] = str(df['device'].iloc[0])
        # Calculate the average duration
        # Convert the DataFrame to a dictionary
        # data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        # msm_data = json.dumps(data_dict)
        # msm_metrics['msm_data'] = msm_data
        return msm_metrics


def run_model_inference(onnx_model_path, input_data_path):
        session = ort.InferenceSession(onnx_model_path)
        # Ensure input matches ONNX model requirements
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_type = session.get_inputs()[0].type
        input_tensor = format_model_input(input_data_path, input_shape, input_type)
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        return outputs


def extract_model(onnx_model_path,node_inputs,node_outputs,model_save_path: str = None) -> None:
    if not os.path.exists(onnx_model_path):
        raise ValueError(f"Invalid input model path: {onnx_model_path}")
    
    if not node_outputs:
        raise ValueError("Output tensor names shall not be empty!")
    
    model = onnx.load(onnx_model_path)
    e = Extractor(model)
    new_model = e.extract_model(node_inputs, node_outputs)
    
    if model_save_path:
        onnx.save(new_model, model_save_path)
    
    return new_model


def split_model(onnx_model_path, json_input, intermediate_outputs, split_group_size, cache_dir, test_inference=True):
    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    exclude_operations = ['Identity',  'Constant']
    all_sub_models = OrderedDict()
    models_with_inputs = OrderedDict()
    
    for idx, node in enumerate(model.graph.node):
        # Skip excluded operations
        if node.op_type in exclude_operations:
            # print(f"Skipping {node.name} of type {node.op_type}...")
            continue
        if node.name in initializers:
            # print(f"{node.name} is an initializer. Skipping...")
            continue
        # print(f"Processing node {node.name} of type {node.op_type}")
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]
        # Save or generate sub-model
        sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)
        all_sub_models[f'split_model_{idx+1}'] = sub_model

    if split_group_size > 1:
        grouped_splits = []
        items = list(all_sub_models.items())
        temp2 = [dict(items[i:i+split_group_size]) for i in range(0, len(items), split_group_size)]
        grouped_splits.extend(temp2)
        all_sub_models = [merge_onnx_models(group) for group in grouped_splits]
    else:
        all_sub_models = list(all_sub_models.values())
    
    for idx, sub_model in enumerate(all_sub_models):
       
        flattened_inputs = []
        for input_tensor in sub_model.graph.input:
            flattened_inputs.append(intermediate_outputs[input_tensor.name].flatten().tolist())
            # flattened_inputs.append(intermediate_outputs[input_tensor.name].numpy()).reshape([-1]).tolist()
        #check if flattened inputs are empty
        if not flattened_inputs:
            continue
        sub_model_name = f'split_model_{idx+1}'

        input_data = {"input_data": flattened_inputs}
    
        sub_model_data_folder = os.path.join(cache_dir, sub_model_name)
        model_save_path = os.path.join(sub_model_data_folder, 'model.onnx')
        input_data_save_path = os.path.join(sub_model_data_folder, 'input.json')
        os.makedirs(sub_model_data_folder, exist_ok=True)
        onnx.save(sub_model, model_save_path)
        with open(input_data_save_path, 'w') as json_file:
            json.dump(input_data, json_file, indent=4)

        # if test_inference:
        #     output = run_model_inference(model_save_path, input_data_save_path)

        models_with_inputs[sub_model_name] = input_data_save_path, model_save_path

    return models_with_inputs

def merge_onnx_models(sub_models:OrderedDict):
    
    # Get the first model from the OrderedDict
    first_model_id, first_model = next(iter(sub_models.items()))
    # base_model = onnx.load(first_model_path)
    merged_model = first_model
    # model_input_data = first_model['input']
    merged_model.graph.ClearField('output')
   
    sub_model_list = list(sub_models.items())
    for idx, (model_id, model) in enumerate(sub_model_list[1:]):
        # sub_model = onnx.load(model_path)
        sub_model = model
        for input_tensor in sub_model.graph.input:
            # Check if the input tensor is already in the merged mode
            name = input_tensor.name
            non_in_mergerd = True
            for merged_input in merged_model.graph.input:
                if merged_input.name == name:
                    non_in_mergerd = False
                    break
            if non_in_mergerd:
                merged_model.graph.input.append(input_tensor) 
            # if input_tensor not in merged_model.graph.input:
            #     merged_model.graph.input.append(input_tensor)
        sub_model.graph.ClearField('input')
        for node in sub_model.graph.node:
            merged_model.graph.node.append(node)
        for initializer in sub_model.graph.initializer:
            merged_model.graph.initializer.append(initializer)

        #remove duplicates from inputs 
        inputs_already_in_merged = set()
        for input_tensor in merged_model.graph.input:
            if input_tensor.name in inputs_already_in_merged:
                merged_model.graph.input.remove(input_tensor)
            else:
                inputs_already_in_merged.add(input_tensor.name)
        #remove duplicates from outputs


        if idx == len(sub_model_list) - 2:  # Last model in the iteration
            for output_tensor in sub_model.graph.output:
                if output_tensor not in merged_model.graph.output:
                    merged_model.graph.output.append(output_tensor)
        for value_info in sub_model.graph.value_info:
            if value_info not in merged_model.graph.value_info:
                merged_model.graph.value_info.append(value_info)
    #look up for the input_data for this model part
    return merged_model
    # return {"model":merged_model, "input": model_input_data}
    
    
def collect_intermediate_inference_outputs(onnx_model_path, input_data_path):
        model = onnx.load(onnx_model_path)
        # Update the model so the final output includes the output of every node, not just the last node
        while len(model.graph.output) > 0: # Remove all existing outputs
            model.graph.output.pop()

        # Perform shape inference to update model with inferred shapes
        shape_info = onnx.shape_inference.infer_shapes(model)

        # Add all intermediate outputs to the graph's outputs
        for node in shape_info.graph.node:
            for output_name in node.output:
                # Ensure the output name is not already in the outputs list
                if not any(o.name == output_name for o in model.graph.output):
                    output_info = onnx.ValueInfoProto()
                    output_info.name = output_name
                    model.graph.output.append(output_info)
        # Initialize InferenceSession with inferred model
        session = ort.InferenceSession(model.SerializeToString())
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_type = session.get_inputs()[0].type

        # Load input data
        input_data = format_model_input(input_data_path, input_shape, input_type)
        
        intermediate_inference_outputs = {}
        intermediate_inference_outputs[input_name] = input_data  #store the input data

        # Run inference for all outputs, including the intermediate outputs
        results = session.run(None, {input_name: input_data})
        # Collect the intermediate inference outputs
        intermediate_inference_outputs = {}
        intermediate_inference_outputs[input_name] = input_data

        # Store results for each output
        for name, result in zip(session.get_outputs(), results):
            intermediate_inference_outputs[name.name] = result
        
        # Display the last item in the dictionary
        # last_key = list(intermediate_inference_outputs.keys())[-]
        # last_value = intermediate_inference_outputs[last_key]
        # print(f"Last key: {last_key}, Last value: {last_value}")
        return intermediate_inference_outputs


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


def count_weights_and_tensors_in_onnx_model(model):
    # Load the ONNX model
    if isinstance(model, str):
        model = onnx.load(model)
    
    total_weights = 0
    total_input_size = 0
    total_output_size = 0

    # Count weights in initializers
    for initializer in model.graph.initializer:
        total_weights += len(onnx.numpy_helper.to_array(initializer).flatten())

    # Count input tensor sizes
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        total_input_size += int(np.prod(shape))

    # Count output tensor sizes
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        total_output_size += int(np.prod(shape))

    return total_weights + total_input_size + total_output_size


class OnnxModelToProve():
    def __init__(self, 
                 job_id, 
                 job_name, 
                 input_data_path,
                 onnx_model_path,
                 num_model_ops=None,
                 num_model_params=None):
        
        self.job_id = job_id
        self.model_name = job_name
        #set data dir to be paretn folder or onxx model file
        self.data_dir = os.path.dirname(onnx_model_path)
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.status = JobStatus.PENDING
        self.num_model_ops = num_model_ops
        self.num_model_params = num_model_params
        self.overwrite = False

        self.vk_path = os.path.join(self.data_dir, 'vk.json')
        self.settings_path = os.path.join(self.data_dir, 'settings.json')
        self.compiled_circuit_path = os.path.join(self.data_dir, 'network.compiled')
        self.pk_path = os.path.join(self.data_dir, 'pk.json')
        self.witness_path = os.path.join(self.data_dir, 'witness.json')
        self.proof_path = os.path.join(self.data_dir, 'proof.pf')

    def generate_model_info(self):
        if self.num_model_ops is None or self.num_model_params is None:
            onnx_model = onnx.load(self.onnx_model_path)
            self.num_model_ops  = len(onnx_model.graph.node)
            self.num_model_params =  0
            for initializer in onnx_model.graph.initializer:
                param_array = onnx.numpy_helper.to_array(initializer)
                self.num_model_params += param_array.size
            
            num_combined_params = count_weights_and_tensors_in_onnx_model(onnx_model)


        #save all model ops as a list in the settings file
        num_model_ops_str = [node.op_type for node in onnx_model.graph.node]
        #get ezkl settings if file exists
        info = {
            "name": self.model_name,
            "onnx_model_path": self.onnx_model_path,
            "input_data_path": self.input_data_path,
            "model_ops": num_model_ops_str,
            "num_model_ops": self.num_model_ops,
            "num_model_params": num_combined_params}
        
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                ezkl_settings = json.load(f)
            info = {**info, **ezkl_settings}
        return info
    
    @time_function
    def _gen_settings(self):
        if not self.overwrite and os.path.isfile(self.settings_path):
            return True
        ezkl.gen_settings(self.onnx_model_path, self.settings_path)
        
    @time_function
    def _calibrate_settings(self):
        ezkl.calibrate_settings(self.input_data_path, self.onnx_model_path, self.settings_path, "resources")

    @time_function
    def _compile_circuit(self):
        if not self.overwrite and os.path.isfile(self.compiled_circuit_path):
            return True
        ezkl.compile_circuit(self.onnx_model_path, self.compiled_circuit_path, self.settings_path)

    @time_function
    def _get_srs(self):
        ezkl.get_srs(self.settings_path)

    @time_function
    def _gen_witness(self):
        ezkl.gen_witness(self.input_data_path, self.compiled_circuit_path, self.witness_path)
        assert os.path.isfile(self.witness_path)
    
    @time_function
    def _setup(self):
        if not self.overwrite and os.path.isfile(self.pk_path):
            logger.info("Skipping setup as key files already exist")
            return True
        
        ezkl.setup(self.compiled_circuit_path, self.vk_path, self.pk_path)

        #rename fft and msms reports so that they are for setup only
        suffix = 'setup'
        for file in 'halo2_ffts.csv', 'halo2_msms.csv':
            if os.path.isfile(file):
                        name, ext = os.path.splitext(file)  # Split filename and extension
                        new_name = f"{name}_{suffix}{ext}"  # Append suffix before extension
                        os.rename(file, new_name)

    @time_function
    def _prove(self):
        logger.info("Starting proof generation")
        ezkl.prove(self.witness_path, self.compiled_circuit_path, self.pk_path, self.proof_path, "single")

        suffix = 'prover'
        for file in 'halo2_ffts.csv', 'halo2_msms.csv':
            if os.path.isfile(file):
                        name, ext = os.path.splitext(file)  # Split filename and extension
                        new_name = f"{name}_{suffix}{ext}"  # Append suffix before extension
                        os.rename(file, new_name)

        assert os.path.isfile(self.proof_path)
    
    @time_function
    def _verify(self):
        try:
            res = ezkl.verify(self.proof_path, self.settings_path, self.vk_path)
            suffix = 'verifier'
            for file in 'halo2_ffts.csv', 'halo2_msms.csv':
                if os.path.isfile(file):
                        name, ext = os.path.splitext(file)  # Split filename and extension
                        new_name = f"{name}_{suffix}{ext}"  # Append suffix before extension
                        os.rename(file, new_name)
            # self.exp_logger.log_value('verified', str(res))
        except Exception as e:
            # logger.exception("Error in verification: %s", e)
            # self.exp_logger.log_value('verified', "False")
            return False
    
    def generate_zk_proof(self, report_dir=None):
        function_times = {}
        functions = [('gen_settings', self._gen_settings),
                ('calibrate_settings', self._calibrate_settings),
                ('compile_circuit', self._compile_circuit),
                ('get_srs', self._get_srs),
                ('witness_gen', self._gen_witness),
                ('setup', self._setup),
                ('prove', self._prove),
                ('verify', self._verify)]
        for func_name, func in functions:
            execution_time = func()
            function_times[f'ezkl_{func_name}(s)'] = f"{execution_time:.3f}"
            logger.info(f"{func_name} took {execution_time:.3f}s")
        return function_times

class GlobalProvingJob():
    def __init__(self, job_name, input_data_path, onnx_model_path, num_of_splits,split_group_size, cache_setup_files=True):
        
        self.model_name = job_name
        self.input_data_path = input_data_path
        self.onnx_model_path = onnx_model_path
        self.num_of_splits = num_of_splits
        self.split_group_size = split_group_size
        self.inference_results = {} #stores results of model inference with and without ZKP proof
        self.status = JobStatus.PENDING
        self.models_to_prove: List[OnnxModelToProve] = []
        date_time_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.cache_directory = os.path.join('cache', self.model_name)
        self.report_directory = os.path.join('reports', self.model_name, date_time_utc_str)
        #create report directory if it does not exist
        os.makedirs(self.report_directory, exist_ok=True)
        self.cache_setup_files = cache_setup_files

    def prepare_for_processing(self, save_ezkl_settings=True):
        
        logger.info(f"Preparing job for processing: {self.model_name}")
        try:
            #check if the model and input data are valid, and store the inference result
            self.inference_results['non_zk_inference'] = run_model_inference(self.onnx_model_path, self.input_data_path)
            #save the inference result to a file
            inference_result_file = os.path.join(self.report_directory, 'inference_results_nonzk.json')
            serializable_output = make_json_serializable(self.inference_results['non_zk_inference'])

            with open(inference_result_file, 'w') as f:
                json.dump(serializable_output, f, indent=4)
        except Exception as e:
            logger.error(f"Error running model inference: {e}. Check the input data and model cmpatibility.")
            raise e
        logger.debug(f" Non-ZK inference output: {self.inference_results['non_zk_inference']}")
        
        if self.num_of_splits is None or self.num_of_splits < 1:

            #copy into cache directory
            os.makedirs(self.cache_directory, exist_ok=True)
            onnx_model_cache_path = os.path.join(self.cache_directory, 'model.onnx')
            input_data_cache_path = os.path.join(self.cache_directory, 'input.json')
            os.system(f'cp {self.onnx_model_path} {onnx_model_cache_path}')
            os.system(f'cp {self.input_data_path} {input_data_cache_path}')


            logger.info(f'No split size provided. Proving the model as a whole')
            self.models_to_prove.append(OnnxModelToProve(job_id=1, 
                                                         job_name=self.model_name,
                                                         input_data_path=input_data_cache_path,
                                                         onnx_model_path=onnx_model_cache_path))
        else:
            logger.info(f'Collecting intermediate inference outputs for sub-models')
            intermediate_inference_outputs = collect_intermediate_inference_outputs(self.onnx_model_path, self.input_data_path)
            logger.info(f'Intermediate inference outputs collected')

            logger.info(f"Splitting model at every node")
            sub_models = split_model(
                self.onnx_model_path, 
                self.input_data_path,
                intermediate_outputs=intermediate_inference_outputs,
                split_group_size=self.split_group_size,
                cache_dir=self.cache_directory)
            
            # if self.split_group_size < len(all_sub_models):
            #     all_sub_models = self.group_models(all_sub_models, self.split_group_size, self.group_split_lists, False)
            
            logger.info(f"total number of sub-models to prove: {len(sub_models)}")
            for idx, (submodel_name) in enumerate(sub_models.keys()):
                submodel_input_path, sub_model_onnx_path = sub_models[submodel_name]
                submodel_name = f"{self.model_name}_{submodel_name}"
                self.models_to_prove.append(OnnxModelToProve(
                    job_id=idx+1,
                    job_name=submodel_name, 
                    input_data_path=submodel_input_path, 
                    onnx_model_path=sub_model_onnx_path))
                
        logger.info("generating settings for proving each model")
        #generate ezkl settings for each model and then summarize the model info in a report
        
        if save_ezkl_settings:
            report_file = os.path.join(self.report_directory, 'ezkl_settings.csv')
            for model in self.models_to_prove:
                #get parent folder of model onnx file
                model._gen_settings()
                info = model.generate_model_info()
                #save model info to report directory

                #create report directory if it does not exist
                os.makedirs(self.report_directory, exist_ok=True)
                file_exists = os.path.isfile(report_file)
                with open(report_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=info.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(info)

    logger.info("Finished preparing job for processing")


    def gen_proof_for_sub_models(self):
        logger.info(f"Generating ZK proof for sub-models")

        ezkl_perf_summary_file = os.path.join(self.report_directory, 'ezkl_perf.csv')
        halo2_perf_summary_file = os.path.join(self.report_directory, 'halo2_perf.csv')

        for idx, model in enumerate (self.models_to_prove):
            logger.info(f"Generating ZK proof for model: {model.model_name} ({idx+1}/{len(self.models_to_prove)})")
           
            model_info = {'name': model.model_name, 'num_ops': model.num_model_ops, 'num_params': model.num_model_params, 'onnx_model_path': model.onnx_model_path}
            ezkl_pref_metrics = model.generate_zk_proof()
            ezkl_perf = {**model_info, **ezkl_pref_metrics}
            
            file_exists = os.path.isfile(ezkl_perf_summary_file)
            with open(ezkl_perf_summary_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=ezkl_perf.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(ezkl_perf)

            circuit_info = read_csv_into_dict('halo2_circuit.csv')
            prover_info = read_csv_into_dict('halo2_prover.csv')

            if os.path.isfile('halo2_ffts_setup.csv'):
                fft_setup_summary = get_fft_summary('halo2_ffts_setup.csv', 'steup')
            if os.path.isfile('halo2_ffts_prover.csv'):
                fft_prover_summary = get_fft_summary('halo2_ffts_prover.csv', 'prove')
            if os.path.isfile('halo2_ffts_verifier.csv'):
                fft_verifier_summary = get_fft_summary('halo2_ffts_verifier.csv', 'verify')

            if os.path.isfile('halo2_msms_setup.csv'):
                msm_setup_summary = get_msm_summary('halo2_msms_setup.csv', 'setup')
            if os.path.isfile('halo2_msms_prover.csv'):
                msm_prover_summary = get_msm_summary('halo2_msms_prover.csv', 'prover')
            if os.path.isfile('halo2_msms_verifier.csv'):
                msm_verifier_summary = get_msm_summary('halo2_msms_verifier.csv', 'verifier')

            #combine all the summaries into one dictionary
            halo2_perf = {**model_info, **circuit_info, **prover_info}
            if fft_setup_summary is not None:
                halo2_perf = {**halo2_perf, **fft_setup_summary}
            if fft_prover_summary is not None:
                halo2_perf = {**halo2_perf, **fft_prover_summary}
            if fft_verifier_summary is not None:
                halo2_perf = {**halo2_perf, **fft_verifier_summary}
            if msm_setup_summary is not None:
                halo2_perf = {**halo2_perf, **msm_setup_summary}
            if msm_prover_summary is not None:
                halo2_perf = {**halo2_perf, **msm_prover_summary}
            if msm_verifier_summary is not None:
                halo2_perf = {**halo2_perf, **msm_verifier_summary}
                
            file_exists = os.path.isfile(halo2_perf_summary_file)
            with open(halo2_perf_summary_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=halo2_perf.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(halo2_perf)

            model_report_dir = os.path.join(self.report_directory, f'{model.model_name}')
            os.makedirs(model_report_dir, exist_ok=True)


            #copy the following files to the report directory
            files_to_copy = ['halo2_ffts_setup.csv','halo2_msms_setup.csv',
                             'halo2_ffts_prover.csv','halo2_msms_prover.csv',
                             'halo2_ffts_verifier.csv','halo2_msms_verifier.csv']
            
            files_to_remove = ['halo2_circuit.csv', 'halo2_prover.csv','halo2_ffts_setup.csv']

            for file in files_to_copy:
                if os.path.isfile(file):
                    os.system(f'mv {file} {model_report_dir}')
            #generate proof for each model

            #remove the files that are not needed
            for file in files_to_remove:
                if os.path.isfile(file):
                    os.remove(file)

        logger.info(f"ZK proof generation completed for all sub-models")

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
        num_of_splits=config.model.split_group_size,
        split_group_size=config.model.split_group_size,
        cache_setup_files=config.cache_setup_files)
    
    # Start system usage logging in a separate process (more isolated than a thread)
    log_file = os.path.join(job.report_directory, "system_usage.log")
    logging_process = multiprocessing.Process(target=log_system_usage, args=(log_file,))
    logging_process.start()
    
    try:
        job.prepare_for_processing(save_ezkl_settings=True)
        job.gen_proof_for_sub_models()

        if not job.cache_setup_files:
            logger.info(f"Cleaning up cache directory: {job.cache_directory}")
            shutil.rmtree(job.cache_directory, ignore_errors=True)

    finally:
        if logging_process.is_alive():
            logging_process.terminate()
            logging_process.join()
        logger.info("All done. Shutting down...")
    
if __name__ == '__main__':
    main()