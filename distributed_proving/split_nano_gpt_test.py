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
from onnx import helper, TensorProto
from oldonnx_split import split_onnx, get_all_cut_points

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
            ezkl_settings = get_ezkl_settings(sub_model)
            counter +=1
            proving_input = {"input_data": inputs}
            sub_model_output_folder = os.path.join(output_folder, f'split_{counter}')
            model_save_path = f'{sub_model_output_folder}/model.onnx'
            input_data_save_path = f'{sub_model_output_folder}/input.json'
            print(f"Processed node {node.name} of type {node.op_type}, num_rows: {ezkl_settings.get('num_rows', 0)}")
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

    return models_with_inputs

def create_subgraph(onnx_model_path, output_folder):
    # Define the inputs and outputs for the subgraph
    node_inputs = ['/Shape_output_0']
    node_outputs = ['/Gather_output_0']

    # Extract the subgraph
    sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)

    # Prepare paths for saving the subgraph and associated files
    sub_model_output_folder = os.path.join(output_folder, 'split_test1')
    model_save_path = os.path.join(sub_model_output_folder, 'model.onnx')

    try:
        # Retrieve and log EZKL settings
        ezkl_settings = get_ezkl_settings(sub_model)
        print(f"num_rows: {ezkl_settings.get('num_rows', 0)}")

        # Ensure the output folder exists
    except Exception as e:
        print(f"Error: {e}")

    # Save the modified subgraph
    os.makedirs(sub_model_output_folder, exist_ok=True)
    onnx.save(sub_model, model_save_path)

def create_submodel_for_node(onnx_model, node_idx, output_folder):
    # Get the node
    node = onnx_model.graph.node[node_idx]

    if node.op_type == "Identity" or node.op_type == "Constant":
        print(f"Skipping {node.op_type} node: {node.name}")
        return  # Skip this node

    # Get all the inputs and outputs associated with the node
    node_inputs = node.input
    node_outputs = node.output
    
    node_inputs = [input_name for input_name in node_inputs if input_name not in [init.name for init in onnx_model.graph.initializer] and 'Constant' not in input_name]

    # Create new graph with just this node
    subgraph_nodes = [node]
    
    # Get the tensors (initializers) required for this node
    subgraph_initializers = [init for init in onnx_model.graph.initializer if init.name in node_inputs or init.name in node_outputs]
    
    # Create a new graph containing only this node and its initializers
    subgraph_graph = helper.make_graph(
        nodes=subgraph_nodes,
        name=f"subgraph_node_{node.name}",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, []) for input_name in node_inputs],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, []) for output_name in node_outputs],
        initializer=subgraph_initializers

    )
    
    # Create a new model with the subgraph graph
    submodel = helper.make_model(subgraph_graph, opset_imports=[onnx.helper.make_opsetid("ai.onnx", 13)],
                                 ir_version=7)  # Set IR version to 13)
    print("Operator Set Version:", submodel.opset_import[0].version)

    try:
        ezkl_settings = get_ezkl_settings(submodel)
        print(f"num_rows: {ezkl_settings.get('num_rows', 0)}")
    except Exception as e:
        print(f"Error: {e}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    submodel_output_path = os.path.join(output_folder, f"submodel_node_{node_idx}.onnx")
    onnx.save(submodel, submodel_output_path)

def split_model(onnx_model_path, output_folder):
    # Load the original ONNX model
    onnx_model = onnx.load(onnx_model_path)
    print("Operator Set Version:", onnx_model.opset_import[0].version)

    # Iterate over each node in the model and create a submodel for it
    for node_idx in range(len(onnx_model.graph.node)):
        create_submodel_for_node(onnx_model, node_idx, output_folder)

# Example usage
onnx_model_path = 'examples/onnx/nanoGPT/network.onnx'
json_input = 'examples/onnx/nanoGPT/input.json'

cut_points = get_all_cut_points(onnx.load(onnx_model_path))
# split_onnx(onnx.load(onnx_model_path), 'tmp')

# split_model(onnx_model_path, 'tmp')

# create_subgraph(onnx_model_path, 'tmp')


# Get intermediate outputs
itermediate_outputs = get_intermediate_outputs(onnx_model_path, json_input)

# Split the model at every node and run inference
models_with_results = split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs)

# models_with_results now contains the sub-models and their inference results
