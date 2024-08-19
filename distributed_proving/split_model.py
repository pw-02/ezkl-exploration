import onnx
import onnxruntime as ort
from onnx import shape_inference, ModelProto
import json
import numpy as np
import os
from onnx.utils import Extractor
from collections import OrderedDict

def flatten_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.reshape([-1]).tolist()  # Flatten and convert to list
        elif isinstance(value, dict):
            d[key] = flatten_ndarray_to_list(value)
    return d

def extract_model(
    input_path: str | os.PathLike,
    input_names: list[str],
    output_names: list[str],
    output_path: str = None,
    check_model: bool = False,
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
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")

    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    onnx.checker.check_model(input_path)
    model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)
    if output_path:
        onnx.save(extracted, output_path)
        if check_model:
            onnx.checker.check_model(output_path)
    return extracted



def load_json_input(input_path, input_shape, input_type, idx = 0):
    input_shape = [-1 if dim == 'batch_size' else dim for dim in input_shape]

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    if input_type == 'tensor(float)':

        input_data = np.array(input_data['input_data'][idx], dtype=np.float32)
        if len(input_shape)>1:
            try:
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
 
    return input_data

def get_intermediate_outputs(onnx_model, json_input):

    model = onnx.load(onnx_model)
    
    #update the the model so the output includes the output of every node, not just the final node
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    shape_info = onnx.shape_inference.infer_shapes(model)   
    for node_output in shape_info.graph.value_info:
        model.graph.output.extend([node_output])
    
    session = ort.InferenceSession(model.SerializeToString())
    input = session.get_inputs()[0]
    input_shape = input.shape
    input_type = input.type
    input_data = load_json_input(json_input, input_shape, input_type)

    # Run inference
    results = session.run(None, {input.name: input_data})
    intermediate_inference_outputs = {}
    # Print intermediate results
    for name, result in zip(session.get_outputs(), results):
        intermediate_inference_outputs[name.name] = result
    return intermediate_inference_outputs

def run_inference_on_onnx_model_using_input_file(model_path, input_file):
    model = onnx.load(model_path)
    session = ort.InferenceSession(model.SerializeToString())
    infer_input = {}
    for idx, model_input in enumerate(session.get_inputs()):
        input_shape = model_input.shape
        input_type = model_input.type
        input_data = load_json_input(input_file, input_shape, input_type, idx)
        infer_input[model_input.name] = input_data
    # input = session.get_inputs()[0]
    # input_shape = input.shape
    # input_type = input.type
    # input_data = load_json_input(input_file, input_shape, input_type)
    # results = session.run(None, {input.name: input_data})
    results = session.run(None, infer_input)
    return results


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
    input_data = []
    # Get the first model from the OrderedDict
    first_model_id, first_model = next(iter(sub_models.items()))
    # base_model = onnx.load(first_model_path)
    merged_model = first_model

    # for input_tensor in merged_model.graph.input:
    #     input_data.append(intermediate_values[input_tensor.name])

    merged_model.graph.ClearField('output')

    sub_model_list = list(sub_models.items())
    for idx, (model_id, model) in enumerate(sub_model_list[1:]):
        # sub_model = onnx.load(model_path)
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
    
    return merged_model

def split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs, output_folder = 'tmp', save_to_file = False):

    models_with_inputs = OrderedDict()

    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    nodes = {}
    parts = []
    for node in model.graph.node:
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]
        if node_inputs and node_outputs:
            nodes[node.name] = (node_inputs, node_outputs) #only want nodes with input/outputs. The others are constants. 
        else:
            pass

    for idx, node_name in enumerate(nodes):
        sub_model_output_folder = os.path.join(output_folder, f'split_{idx+1}')

        model_save_path = f'{sub_model_output_folder}/model.onnx'
        input_data_save_path = f'{sub_model_output_folder}/input.json'
        # model_save_path = f'{sub_model_output_folder}/split_{idx+1}_model.onnx'
        # input_data_save_path = f'{sub_model_output_folder}/split_{idx+1}_input.json'

        # print(f"Processing Split {idx+1}, Node Name: {node_name}")
        node_inputs, node_outputs = nodes[node_name]
        if save_to_file:
            os.makedirs(sub_model_output_folder, exist_ok=True)
            sub_model = extract_model(onnx_model_path, node_inputs, node_outputs,model_save_path)
        else:
            sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)

        session = ort.InferenceSession(sub_model.SerializeToString())

        #the inputs to each node are the outputs of all parent nodes. 
        # since the first node has no parent, we add it manually to 'itermediate_outputs'

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
        
        # if save_to_file:
        #     models_with_inputs[f'split_model_{idx+1}'] = {"model": model_save_path, "input": input_data_save_path}
            # models_with_inputs.append((model_save_path,input_data_save_path))
        # else:
            # models_with_inputs.append((sub_model,proving_input))
        models_with_inputs[f'split_model_{idx+1}'] = sub_model

    return models_with_inputs

if __name__ == "__main__":
    models_to_test = [
        # ('examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx', 'examples/onnx/mobilenet/input.json'),
        # ('examples/onnx/mnist_gan/network.onnx', 'examples/onnx/mnist_gan/input.json')
        # ('examples/onnx/nanoGPT/network.onnx', 'examples/onnx/nanoGPT/input.json'),
        ( 'examples/onnx/mnist_classifier/network.onnx', 'examples/onnx/mnist_classifier/input.json'),

    ]

    for onnx_file, input_file in models_to_test:

        full_model_result = run_inference_on_onnx_model(onnx_file, input_file)

        # Get the output tensor(s) of every node in the model during inference
        intermediate_results = get_intermediate_outputs(onnx_file, input_file)
        n_parts = np.inf
        split_onnx_model(onnx_file, input_file,  intermediate_results,n_parts, f'examples/split_models/mnist_classifier', True)  
        #result  = split_onnx_model(onnx_file, input_file,  intermediate_results,n_parts)  

