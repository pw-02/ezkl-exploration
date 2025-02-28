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
def flatten_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.reshape([-1]).tolist()  # Flatten and convert to list
        elif isinstance(value, dict):
            d[key] = flatten_ndarray_to_list(value)
    return d

def extract_model(
    onnx_model_path,
    node_inputs,
    node_outputs,
    model_save_path: str = None,
) -> None:
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Arguments:
        input_path (str | os.PathLike): The path to original ONNX model.
        output_path (str | os.PathLike): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
        check_model (bool): Whether to run model checker on the extracted model.
    """
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



def load_json_input(input_path, input_shape, input_type, idx = 0):
    input_shape = [-1 if dim == 'batch_size' else dim for dim in input_shape]

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    if input_type == 'tensor(float)':

        input_data = np.array(input_data['input_data'][idx], dtype=np.float32)
        if len(input_shape)>1:
            try:
                # input_shape.remove('N')
                input_data = input_data.reshape(input_shape)
            except ValueError as e:

                raise ValueError(f"Input data cannot be reshaped from shape {input_data.shape} to the expected shape {input_shape}: {e}")

    elif input_type == 'tensor(int64)': 
        if len(input_shape)>0:
            input_data = np.array(input_data['input_data'][idx], dtype=np.int64)
        else:
            input_data = np.array(input_data['input_data'][0][0],dtype=np.int64) 

    elif input_type == 'tensor(int32)':
        return input_data
    
    # Try to reshape the input data to match the expected shape
    # try:
    #     input_data = input_data.reshape(input_shape)
    # except ValueError as e:
    #     raise ValueError(f"Input data cannot be reshaped from shape {input_data.shape} to the expected shape {input_shape}: {e}")
 

    # # Check if the input data matches the expected shape
    # if list(input_data.shape) != input_shape:
    #     raise ValueError(f"Input data shape {input_data.shape} does not match the expected shape {input_shape}")
    if 'nanoGPT' in input_path:
        input_data = np.reshape(input_data, (1, 64))  # Shape: (1, 64)

    return input_data

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


def run_inference_on_onnx_model(model_path, itermediate_values):
    model = onnx.load(model_path)
    session = ort.InferenceSession(model.SerializeToString())
    infer_input = {}
    for idx, model_input in enumerate(session.get_inputs()):
        infer_input[model_input.name] = itermediate_values[model_input.name]
    results = session.run(None, infer_input)
    return results


def transform_nodes_into_splits(node_info, parts_count, debug = False):
    node_names = list(node_info.keys())
    total_nodes = len(node_names)
    part_size = total_nodes // parts_count
    parts = []
    for i in range(parts_count):
        start_index = i * part_size
        # Ensure the last part includes all remaining nodes
        end_index = (i + 1) * part_size if i < parts_count - 1 else total_nodes
        part_names = node_names[start_index:end_index]
        part = [(name, node_info[name][0], node_info[name][1]) for name in part_names]
        parts.append(part)    
    if debug:
        # Print the nodes in each part
        for i, part in enumerate(parts):
            print(f"Part {i+1}:")
            for info in part:
                print(f"Node Name: {info[0]}, Input: {info[1]}, Output: {info[2]}")
            print()
    return parts

def remove_dups(original_list):
    seen = set()
    return [item for item in original_list if not (item in seen or seen.add(item))]


def split_onnx_model(onnx_model_path, json_input, itermediate_outputs, n_parts = np.inf, output_folder = 'tmp', save_to_file = False):
    models_with_inputs = []
    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    node_info = {} #contains the input and output of each node
    nodes_that_dont_have_inout_and_output =[]
    for node in model.graph.node:
        if 'Shape' ==  node.op_type:
            pass
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]
        # node_shapes = [shape for shape in node.shape if shape not in initializers and 'Constant' not in shape]

        if node_inputs and node_outputs:
            node_info[node.name] = (node_inputs, node_outputs)
        else:
            nodes_that_dont_have_inout_and_output.append(node.name)
    
    n_parts = min(len(node_info), n_parts)
    splits = transform_nodes_into_splits(node_info, n_parts)
    if save_to_file:
        os.makedirs(output_folder, exist_ok=True)
    
    for idx, current_split in enumerate(splits):
        sub_model_output_folder = os.path.join(output_folder, f'split_{idx+1}')
        #if save_to_file:
        os.makedirs(sub_model_output_folder, exist_ok=True)
        model_save_path = f'{sub_model_output_folder}/model.onnx'
        input_data_save_path = f'{sub_model_output_folder}/input.json'
        
        first_node_in_current_split = node_info[current_split[0][0]] #get inputput/ouput of the first node in the current split
        input_names = [name for name in first_node_in_current_split[0]] #get input names of the first node in the current split
        split_outputs = set()
        for node in current_split:
            node_name = node[0]
            split_outputs.update(node_info[node_name][1])
        
        # Check if any node has an input that is not an output of another node in the part
        for node in current_split:
            node_name = node[0]
            for input_name in node_info[node_name][0]:
                if input_name not in split_outputs:
                    if input_name not in input_names:
                        input_names.append(input_name)
        
        last_node_in_split = node_info[current_split[-1][0]]
        output_names = [name for name in last_node_in_split[1]]
        input_names=  remove_dups(input_names)
        output_names= remove_dups(output_names)
        if save_to_file:
            sub_model = extract_model(onnx_model_path, input_names, output_names,model_save_path)
        else:
            sub_model = extract_model(onnx_model_path, input_names, output_names)
        
        session = ort.InferenceSession(sub_model.SerializeToString())
        
        input_names = [input.name for input in session.get_inputs()]
        if idx == 0: #first part takes in the inital input
            input = session.get_inputs()[0]
            input_shape = input.shape
            input_type = input.type
            input_data = load_json_input(json_input, input_shape, input_type)
            itermediate_outputs[input.name] = input_data
        assert all(name in itermediate_outputs for name in input_names), "Input data dictionary keys must match the model input names."
        # inference_input = {}
        # for name in input_names:
        #     inference_input[name] = itermediate_outputs[name] 
        # results = session.run(None, inference_input)
        # # print(f"Inference results for {node_name}:", results)

        inputs =  []
        for name in input_names:
            inputs.append(itermediate_outputs[name].flatten().tolist())
        proving_input = {"input_data": inputs}
        
        if save_to_file:
            with open(input_data_save_path, 'w') as json_file:
                json.dump(proving_input, json_file, indent=4)
        
        if save_to_file:
            models_with_inputs.append((model_save_path,input_data_save_path))
        else:
            models_with_inputs.append((sub_model,proving_input))

    return models_with_inputs


def merge_onnx_models(sub_models:OrderedDict):
    
    # Get the first model from the OrderedDict
    first_model_id, first_model = next(iter(sub_models.items()))
    # base_model = onnx.load(first_model_path)
    merged_model = first_model['model']
    model_input_data = first_model['input']
    merged_model.graph.ClearField('output')
   
    sub_model_list = list(sub_models.items())
    for idx, (model_id, model) in enumerate(sub_model_list[1:]):
        # sub_model = onnx.load(model_path)
        sub_model = model['model']
        for input_tensor in sub_model.graph.input:
            if input_tensor not in merged_model.graph.input:
                merged_model.graph.input.append(input_tensor)
        sub_model.graph.ClearField('input')
        for node in sub_model.graph.node:
            merged_model.graph.node.append(node)
        for initializer in sub_model.graph.initializer:
            merged_model.graph.initializer.append(initializer)

        if idx == len(sub_model_list) - 2:  # Last model in the iteration
            for output_tensor in sub_model.graph.output:
                if output_tensor not in merged_model.graph.output:
                    merged_model.graph.output.append(output_tensor)
        for value_info in sub_model.graph.value_info:
            if value_info not in merged_model.graph.value_info:
                merged_model.graph.value_info.append(value_info)
    #look up for the input_data for this model part
    
    return {"model":merged_model, "input": model_input_data}
    
def merge_onnx_models(sub_models:OrderedDict):
    combined_node_indices = []
    
    # Get the first model from the OrderedDict
    first_model_id, first_model = next(iter(sub_models.items()))
    combined_node_indices.append(int(first_model_id.split('_')[-1])  # Split by underscore and take the last part
)

    if len(sub_models) == 1:
        return first_model, combined_node_indices
    
    merged_model = copy.deepcopy(first_model)

    # for input_tensor in merged_model.graph.input:
    #     input_data.append(intermediate_values[input_tensor.name])

    merged_model.graph.ClearField('output')

    sub_model_list = list(sub_models.items())
    for idx, (model_id, model) in enumerate(sub_model_list[1:]):
        # sub_model = onnx.load(model_path)
        model = copy.deepcopy(model)
        combined_node_indices.append(int(model_id.split('_')[-1]))  # Split by underscore and take the last part
        sub_model = model
        for input_tensor in sub_model.graph.input:
                #also check it is not an output of any node
                if not any(input_tensor.name == node_output for node in merged_model.graph.node for node_output in node.output):
                    merged_model.graph.input.append(input_tensor)
                    # input_data.append(intermediate_values[input_tensor.name])

        sub_model.graph.ClearField('input')
        for node in sub_model.graph.node:
            merged_model.graph.node.append(node)
        for initializer in sub_model.graph.initializer:
            merged_model.graph.initializer.append(initializer)

        if idx == len(sub_model_list) - 2:  # Last model in the iteration
            for output_tensor in sub_model.graph.output:
                if output_tensor not in merged_model.graph.output:
                    merged_model.graph.output.append(output_tensor)
        for value_info in sub_model.graph.value_info:
            if value_info not in merged_model.graph.value_info:
                merged_model.graph.value_info.append(value_info)
    #look up for the input_data for this model part
    
    return merged_model, combined_node_indices

def split_onnx_model_at_every_node(onnx_model_path, json_input, intermediate_outputs, output_folder = 'tmp', save_to_file = True):
    models_with_inputs = OrderedDict()
    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    exclude_operations = ['Identity',  'Constant']
    counter =0
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
        session = ort.InferenceSession(sub_model.SerializeToString())
        input_names = [input.name for input in session.get_inputs()]
        if counter == 0:  # First part takes in the initial input
            input_names = input_names[0]
            input = session.get_inputs()[0]
            input_shape = input.shape
            input_type = input.type
            input_data = load_json_input(json_input, input_shape, input_type)
            intermediate_outputs[input.name] = input_data
        inputs = []
        for name in input_names:
            inputs.append(intermediate_outputs[name].flatten().tolist())
        counter +=1
        proving_input = {"input_data": inputs}
        # current_node_inputs.clear() 
        if save_to_file:
            sub_model_output_folder = os.path.join(output_folder, f'split_{counter}')
            model_save_path = f'{sub_model_output_folder}/model.onnx'
            input_data_save_path = f'{sub_model_output_folder}/input.json'
            os.makedirs(sub_model_output_folder, exist_ok=True)
            onnx.save(sub_model, model_save_path)
            with open(input_data_save_path, 'w') as json_file:
                json.dump(proving_input, json_file, indent=4)
        # current_node_inputs = node_outputs
        models_with_inputs[f'split_model_{counter}'] = sub_model
    return models_with_inputs



# def split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs, output_folder = 'tmp', save_to_file = True):

#     models_with_inputs = OrderedDict()

#     model = onnx.load(onnx_model_path)
#     initializers = {init.name for init in model.graph.initializer}
#     nodes = {}
#     parts = []

#     for idx, node in enumerate(model.graph.node):
        
#         if node.name in initializers:
#             print(node.name + " is an initializer. Skipping...")
#             continue

#         if node.op_type == 'Identity':
#             print(node.name + " is an Identity node. Skipping...")
#             continue

#         print(node.name, node.op_type)
#         node_inputs = [input for input in node.input]
#         node_outputs = [output for output in node.output]
        
#         sub_model_output_folder = os.path.join(output_folder, f'split_{idx+1}')
#         model_save_path = f'{sub_model_output_folder}/model.onnx'
#         input_data_save_path = f'{sub_model_output_folder}/input.json'
#         if save_to_file:
#             os.makedirs(sub_model_output_folder, exist_ok=True)
#             sub_model = extract_model(onnx_model_path, node_inputs, node_outputs,model_save_path)
#         else:
#             sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)

#         session = ort.InferenceSession(sub_model.SerializeToString())

#         #the inputs to each node are the outputs of all parent nodes. 
#         # since the first node has no parent, we add it manually to 'itermediate_outputs'
#         input_names = [input.name for input in session.get_inputs()]
#         if idx == 0: #first part takes in the inital input
#             input = session.get_inputs()[0]
#             input_shape = input.shape
#             input_type = input.type
#             input_data = load_json_input(json_input, input_shape, input_type)
#             itermediate_outputs[input.name] = input_data

#         # assert all(name in itermediate_outputs for name in input_names), "Input data dictionary keys must match the model input names."
#         # inference_input = {}
#         # for name in input_names:
#         #     inference_input[name] = itermediate_outputs[name] 
#         # results = session.run(None, inference_input)
#         # # print(f"Inference results for {node_name}:", results)

#         inputs =  []
#         # if node_name == '/Gather':
#         #      inputs.append(['batch_size', 64])
#         # else:
#         # for name in input_names:
#         #     inputs.append(itermediate_outputs[name].flatten().tolist())
#         for name in input_names:
#             inputs.append(itermediate_outputs[name].flatten().tolist())
#         proving_input = {"input_data": inputs}
#         if save_to_file:
#             with open(input_data_save_path, 'w') as json_file:
#                 json.dump(proving_input, json_file, indent=4)
        
#         # if save_to_file:
#         #     models_with_inputs[f'split_model_{idx+1}'] = {"model": model_save_path, "input": input_data_save_path}
#             # models_with_inputs.append((model_save_path,input_data_save_path))
#         # else:
#             # models_with_inputs.append((sub_model,proving_input))
#         models_with_inputs[f'split_model_{idx+1}'] = sub_model

#     return models_with_inputs




def analzuye():
    from onnx import helper

    model_1 = onnx.load('examples/onnx/nanoGPT/network.onnx')
    # model_2 = onnx.load('tmp/split_2/model.onnx')
    output_file = 'gptnode_sizes.txt'

    # Ensure inputs are properly defined
  # Iterate over the nodes and print the size of each
    with open(output_file, 'w') as f:
        # Iterate over the nodes and write the size of each to the file
        for i, node in enumerate(model_1.graph.node):
            num_inputs = len(node.input)
            num_outputs = len(node.output)
            f.write(f"Node #{i}: {node.name}\n")
            f.write(f"  Operation: {node.op_type}\n")
            f.write(f"  Number of Inputs: {num_inputs}\n")
            f.write(f"  Number of Outputs: {num_outputs}\n")
            f.write("-" * 40 + "\n")

    # onnx.save(model_2, "modified_model.onnx")
    # pass

if __name__ == "__main__":
    models_to_test = [
        # ('examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx', 'examples/onnx/mobilenet/input.json'),
        # ('examples/onnx/mnist_gan/network.onnx', 'examples/onnx/mnist_gan/input.json')
        #  ('examples/onnx/nanoGPT/network.onnx', 'examples/onnx/nanoGPT/input.json'),
        # ( 'examples/onnx/mnist_classifier/network.onnx', 'examples/onnx/mnist_classifier/input.json'),
        # ('examples/onnx/lenet_5/network.onnx', 'examples/onnx/lenet_5/input.json'),
        ('examples/onnx/resnet18/shufflenet-v2-12-qdq.onnx', 'examples/onnx/resnet18/data.json'),


    ]
    # analzuye()
    # for onnx_file, input_file in models_to_test:

    #     full_model_result = run_inference_on_onnx_model_using_input_file(onnx_file, input_file)
    #     print(f"Full Model Inference Result: {full_model_result}")
    #     # Get the output tensor(s) of every node in the model during inference
    #     intermediate_results = get_intermediate_outputs(onnx_file, input_file)
    #     n_parts = np.inf
    #     # split_onnx_model(onnx_file, input_file,  intermediate_results,n_parts, f'tmp', True)  
    #     #result  = split_onnx_model(onnx_file, input_file,  intermediate_results,n_parts)  

    #     all_sub_models = split_onnx_model_at_every_node(onnx_file, input_file, intermediate_results, save_to_file=False)

    #     pass