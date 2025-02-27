import os
import json
import onnx
import onnxruntime as ort
from collections import OrderedDict
import numpy as np
import json
from onnx.utils import Extractor
import ezkl
import os
import shutil
from distributed_proving.usefulutils import analyze_onnx_model_for_zk_proving
import csv
from  distributed_proving.split_model import get_intermediate_outputs, split_onnx_model_at_every_node
import math
import glob
from examples.onnx.nanoGPT.gen import GPTConfig, GPT
import torch

def load_json_input(json_input, input_shape, input_type):
    # Logic to load and preprocess input data from the provided JSON
    with open(json_input, 'r') as f:
        data = json.load(f)
        input_data = data['input_data'][0]
    
    # Convert input data to numpy array based on the type
    if input_type == 'tensor(float)':
        input_data = np.array(input_data, dtype=np.float32)
    elif input_type == 'tensor(int64)':
        input_data = np.array(input_data, dtype=np.int64)
    elif input_type == 'tensor(int32)':
        input_data = np.array(input_data, dtype=np.int32)
    input_data = input_data.reshape(1, 64)  # Shape: (1, 64)
    # You can add preprocessing logic here if needed, like reshaping or converting types
    return input_data


def extract_model(onnx_model_path, node_inputs, node_outputs, model_save_path=None) -> None:
    model = onnx.load(onnx_model_path)
    e = Extractor(model)
    new_model = e.extract_model(node_inputs, node_outputs)

    if model_save_path:
        onnx.save(new_model, model_save_path)
        
    return new_model


def main():
    combinations_to_test = [[4,64], [5,64], [10,64],[15,64],
                            [20,64], [25,64], [4,80], [4,96],
                            [4,112], [4,128], [4,144]]
    json_input = 'examples/onnx/nanoGPT/input.json'
    onnx_model_folder = 'examples/onnx/nanoGPT'
    log_folder = 'logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    for n_layer, n_embd in combinations_to_test:
        print(f"Processing gpt n_layer: {n_layer}, n_embd: {n_embd}")
        total_splits = 0
        total_onnx_parmeters = 0
        total_rows = 0
        total_assignments = 0

        gptconf = GPTConfig(block_size=64, vocab_size=65, n_layer=n_layer,
                    n_head=4, n_embd=n_embd, dropout=0.0, bias=False)
        model = GPT(gptconf)
        torch_global_model_num_params = model.get_num_params()
        x = torch.randint(65, (1, 64))
        torch_out = model(x)
        model_output_path = f"{onnx_model_folder}/nano_gpt_{n_layer}_layers_{n_embd}_embd.onnx"
        torch.onnx.export(model, x, model_output_path,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=13,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],   # the model's input names
                        output_names=['output']  # the model's output names
        )
        model_info, ezkl_settings = analyze_onnx_model_for_zk_proving(model_output_path)
        global_model_total_rows =ezkl_settings['num_rows']
        onnx_global_model_num_params = model_info['num_model_params']

        
        model_base_name = os.path.basename(model_output_path).split('.')[0]
        # Get intermediate outputs
        itermediate_outputs = get_intermediate_outputs(model_output_path, json_input)

          # Split the model at every node and run inference
        all_sub_models = split_onnx_model_at_every_node(model_output_path, json_input, itermediate_outputs,'tmp',False)
        # print(f"Number of sub-models: {len(all_sub_models)}")
        # print(f"Number of sub-models: {len(all_sub_models)}")
        ezkl_settings_path = f'{log_folder}/ezkl_{model_base_name}_settings.csv'
        #delete the file if it exists
        if os.path.exists(ezkl_settings_path):
            os.remove(ezkl_settings_path)

        for idx, (model_name, model) in enumerate(all_sub_models.items()):
            node_name = model.graph.node[-1].name
            # print(f"Processing {model_name}, Node name: {node_name}")
            report_info = {
                'node_idx': idx+1,
                'model_name': model_name,
                'node_name': node_name,
            }
            model_info, ezkl_settings = analyze_onnx_model_for_zk_proving(model)
            report_info['num_model_ops'] = model_info['num_model_ops']
            report_info['num_model_params'] = int(model_info['num_model_params'])
            report_info['num_model_constants'] = model_info['num_model_constants']
            report_info['logrows'] = ezkl_settings['run_args']['logrows']
            report_info.update(ezkl_settings)
            report_info.pop('run_args')
        
          
            file_exists = os.path.isfile(ezkl_settings_path)
            with open(ezkl_settings_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=report_info.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(report_info)
            total_splits += 1
            total_onnx_parmeters += model_info['num_model_params']
            total_model_constants = model_info['num_model_constants']
            total_rows += model_info['zk_circuit_num_rows']
            total_assignments += model_info['zk_circuit_num_assignments']
        
        # Find the smallest power of 2 greater than or equal to the number
        log2rows = math.ceil(math.log2(total_rows))
        print("=========================================")

        summary_info = {
            'model_name': model_output_path,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'global_model_num_params (torch)': torch_global_model_num_params,
            'global_model_num_params (onnx)': onnx_global_model_num_params,
            'global_model_num_circut_rows': global_model_total_rows,
            'global_model_num_circut_logrows': math.ceil(math.log2(global_model_total_rows)),
            'total_sub_models': len(all_sub_models),
            'split_models_num_params (onnx)': total_onnx_parmeters,
            'split_models_num_constants': total_model_constants,
            'split_models_num_circut_rows': total_rows,
            'split_models_num_circut_logrows': log2rows,
            'split_models_num_assignments': total_assignments
        }
        print(summary_info)
        summary_path = f'{log_folder}/ezkl_nanogpt_summary_settings.csv'
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=summary_info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_info)


# def mian():

#     onnx_model_folder = 'examples/onnx/nanoGPT'
#     json_input = 'examples/onnx/nanoGPT/input.json'
#     onnx_files = glob.glob(f"{onnx_model_folder}/*.onnx")

#     for onnx_model_path in onnx_files:
#         total_splits = 0
#         total_parmeters = 0
#         total_rows = 0
#         total_assignments = 0

#         print(f"Processing: {onnx_model_path}")
#         model_base_name = os.path.basename(onnx_model_path).split('.')[0]
#         # Get intermediate outputs
#         itermediate_outputs = get_intermediate_outputs(onnx_model_path, json_input)

#         # Split the model at every node and run inference
#         all_sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs,'tmp',False)
#         # print(f"Number of sub-models: {len(all_sub_models)}")
#         print(f"Number of sub-models: {len(all_sub_models)}")
#         for idx, (model_name, model) in enumerate(all_sub_models.items()):
#             node_name = model.graph.node[-1].name
#             # print(f"Processing {model_name}, Node name: {node_name}")
#             report_info = {
#                 'node_idx': idx+1,
#                 'model_name': model_name,
#                 'node_name': node_name,
#             }
#             model_info, ezkl_settings = analyze_onnx_model_for_zk_proving(model)
#             report_info['num_model_ops'] = model_info['num_model_ops']
#             report_info['num_model_params'] = int(model_info['num_model_params'])
#             report_info['num_model_constants'] = model_info['num_model_constants']
#             report_info['logrows'] = ezkl_settings['run_args']['logrows']
#             report_info.update(ezkl_settings)
#             report_info.pop('run_args')
#             log_folder = 'logs'
#             if not os.path.exists(log_folder):
#                 os.makedirs(log_folder)
#             ezkl_settings_path = f'{log_folder}/ezkl_{model_base_name}_settings.csv'
#             file_exists = os.path.isfile(ezkl_settings_path)
#             with open(ezkl_settings_path, mode='a', newline='') as file:
#                 writer = csv.DictWriter(file, fieldnames=report_info.keys())
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(report_info)
#             total_splits += 1
#             total_parmeters += model_info['num_model_params']
#             total_model_constants = model_info['num_model_constants']
#             total_rows += model_info['zk_circuit_num_rows']
#             total_assignments += model_info['zk_circuit_num_assignments']
        
#         # Find the smallest power of 2 greater than or equal to the number
#         log2rows = math.ceil(math.log2(total_rows))

#         print(
#             f"Total parameters: {total_parmeters}",
#             f"Total model constants: {total_model_constants}",
#             f"Total rows: {total_rows}",
#             f"Logrows: {log2rows}",
#             f"Total assignments: {total_assignments}", sep='\n')
#     # models_with_results now contains the sub-models and their inference results

if __name__ == '__main__':
    main()