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
from utils import analyze_onnx_model_for_zk_proving
import csv

def get_ezkl_settings(onnx_model):
    """Generate and return EZKL settings."""
    temp_dir = os.path.join('ezkl_settings', '1')
    os.makedirs(temp_dir, exist_ok=True)
    settings_path = os.path.join(temp_dir, 'settings.json')

    # Generate the EZKL settings if they don't exist
    
    if not os.path.exists(settings_path):
        if isinstance(onnx_model, str):
            ezkl.gen_settings(onnx_model, settings_path)
        else:
            onnx_model_path = os.path.join(temp_dir, 'model.onnx')
            onnx.save(onnx_model, onnx_model_path)
            ezkl.gen_settings(onnx_model_path, settings_path)
        # Read the settings from the generated file
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)
    # except Exception as e:
    #     print(f"Error: {e}")
    #     settings_data = {}

    # Cleanup the temporary directory if requested
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    return settings_data



def get_intermediate_outputs(onnx_model, json_input):
    model = onnx.load(onnx_model)

    # Update the model so the final output includes the output of every node, not just the last node
    while len(model.graph.output) > 0:
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
    input_data = load_json_input(json_input, input_shape, input_type)

    # Run inference for the initial input to get all intermediate outputs
    intermediate_inference_outputs = {}
    intermediate_inference_outputs[input_name] = input_data

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



def split_onnx_model_at_every_node(onnx_model_path, 
                                            json_input, 
                                            intermediate_outputs, 
                                            output_folder='tmp', 
                                            save_to_file=True):
    models_with_inputs = OrderedDict()
    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    nodes = {}
    parts = []

    # Define operation types to exclude from the split
    exclude_operations = ['Identity', 'Constant', 'Dropout', 'Reshape', 'Clip', 'Squeeze', 'Unsqueeze', 'Transpose', 'Shape', 'Gather','Cast']
    exclude_operations.clear()
    exclude_operations = ['Identity',  'Constant']
    counter =0
    # current_node_inputs = []

    for idx, node in enumerate(model.graph.node):
        # Skip excluded operations
        if node.op_type in exclude_operations:
            # print(f"Skipping {node.name} of type {node.op_type}...")
            continue

        if node.name in initializers:
            print(f"{node.name} is an initializer. Skipping...")
            continue

        print(f"Processing node {node.name} of type {node.op_type}")
        # Filter out 'Constant' nodes from node.inputs and node.outputs
        # if current_node_inputs:
        #     node_inputs = current_node_inputs
        # else:
        #     node_inputs  =[]
     
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]

        # Save or generate sub-model
        sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)
        session = ort.InferenceSession(sub_model.SerializeToString())
        input_names = [input.name for input in session.get_inputs()]
        if counter == 0:  # First part takes in the initial input
            input_names = ['input']
            input = session.get_inputs()[0]
            input_shape = input.shape
            input_type = input.type
            input_data = load_json_input(json_input, input_shape, input_type)
            intermediate_outputs[input.name] = input_data
        inputs = []
        for name in input_names:
            inputs.append(intermediate_outputs[name].flatten().tolist())

  
        try:
            model_info = analyze_onnx_model_for_zk_proving(sub_model)
            # ezkl_settings = get_ezkl_settings(sub_model)
            counter +=1
            proving_input = {"input_data": inputs}
            sub_model_output_folder = os.path.join(output_folder, f'split_{counter}')
            model_save_path = f'{sub_model_output_folder}/model.onnx'
            input_data_save_path = f'{sub_model_output_folder}/input.json'
            print(f"Processed node {node.name} of type {node.op_type}, num_rows: {model_info.get('zk_circuit_num_rows', 0)}")
            # current_node_inputs.clear() 
            if save_to_file:
                os.makedirs(sub_model_output_folder, exist_ok=True)
                onnx.save(sub_model, model_save_path)
                with open(input_data_save_path, 'w') as json_file:
                    json.dump(proving_input, json_file, indent=4)
            # current_node_inputs = node_outputs
            models_with_inputs[f'split_model_{counter}'] = sub_model
          
        except Exception as e:
            print(f"Error: {e}")
            # if not current_node_inputs:
            #     current_node_inputs = node_inputs
            # current_node_inputs = node_inputs

        # print(f"num_rows {node.name}: {ezkl_settings.get('num_rows', 0)}")

        #save settings to csv and have first two coumns being node index and node name

        log_folder = 'logs'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        ezkl_settings_path = f'{log_folder}/ezkl_nanoGPT_settings.csv'
        model_info['node_name'] = node.name
        model_info['node_idx'] = counter
        file_exists = os.path.isfile(ezkl_settings_path)
        with open(ezkl_settings_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=model_info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(model_info)



    return models_with_inputs



# Example usage
onnx_model_path = 'examples/onnx/nanoGPT/network.onnx'
json_input = 'examples/onnx/nanoGPT/input.json'


# Get intermediate outputs
itermediate_outputs = get_intermediate_outputs(onnx_model_path, json_input)

# Split the model at every node and run inference
models_with_results = split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs)

# models_with_results now contains the sub-models and their inference results
