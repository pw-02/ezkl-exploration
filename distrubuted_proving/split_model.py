import onnx
import onnxruntime as ort
from onnx import helper
import json
import numpy as np
from typing import List
from onnx import shape_inference, ModelProto
from onnx.utils import Extractor
import os
import copy

def load_json_input(input_path):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    # Convert to NumPy array and reshape to (1, 3, 224, 224)
    if 'net' in input_path:
        input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is float32
        if input_data.size != 3 * 224 * 224:
            raise ValueError(f"Input data must be of size {3 * 224 * 224}, but got {input_data.size}")
        input_data = input_data.reshape(1, 3, 224, 224)
        return input_data
    elif 'mnist_gan' in input_path or 'little_transformer' in input_path:
        input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is floa
        return input_data
    elif 'mnist_classifier' in input_path:
        input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is floa
        if input_data.size != 1 * 28 * 28:
            raise ValueError(f"Input data must be of size {1 * 28 * 28}, but got {input_data.size}")
        input_data = input_data.reshape(1, 1, 28, 28)

    if input_data is None:
        raise ValueError(f"Input data is None")
    if 'input_data' in input_data:
        return input_data['input_data']
    else:
        return input_data


def run_inference(model_path, input_file):
    model = onnx.load(model_path)
    input_data = load_json_input(input_file)
    session = ort.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: input_data})

    return results

def get_intermediate_outputs(onnx_model, json_input):
    model = onnx.load(onnx_model)
    nodes = model.graph.node
    all_node_outputs = []
    shape_info = shape_inference.infer_shapes(model)

    # Collect all node outputs
    for node in nodes:
        all_node_outputs.extend([output for output in node.output if '/' in output])

    shape_info = onnx.shape_inference.infer_shapes(model)   
    errors = []
    ok = []
    for idx,i_out in enumerate(shape_info.graph.value_info):
        try:
            model.graph.output.extend([i_out])
            session = ort.InferenceSession(model.SerializeToString())
            ok.append((idx, i_out.name))
        except Exception as e:
            errors.append((idx, i_out.name, str(e)))
    if errors:
        print("Errors occurred while extending model outputs:")
        for error in errors:
            print(f"Error at index {error[0]} with output name '{error[1]}': {error[2]}")

    # Load and preprocess the JSON input
    input_data = load_json_input(json_input)

    # Run inference
    input_name = session.get_inputs()[0].name
    
    results = session.run(None, {input_name: input_data})
    
    intermediate_inference_outputs = {}
    # Print intermediate results
    for name, result in zip(session.get_outputs(), results):
        intermediate_inference_outputs[name.name] = result
    return intermediate_inference_outputs

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

def divide_nodes_into_parts(node_info, parts_count, debug = False):
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

def run_inference_on_split_model(onnx_model, json_input, itermediate_outputs, n_parts = np.inf):
    model = onnx.load(onnx_model)
    nodes = model.graph.node
    node_info = {}
    initializers = {init.name for init in model.graph.initializer}
    nodes_that_dont_have_inout_and_output =[]
    node_info = {}
    for node in nodes:
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]
        if node_inputs and node_outputs:
            node_info[node.name] = (node_inputs, node_outputs)
        else:
            nodes_that_dont_have_inout_and_output.append(node.name)

    n_parts = min(len(node_info), n_parts)

    parts = divide_nodes_into_parts(node_info, n_parts)

    model_parts = []
    output_paths = [f"test_parts/part{i+1}_model.onnx" for i in range(n_parts)]

    # output_paths = [f"model_to_circuit_relationship/mnistgan_x_splits/onnx_models/part{i+1}_model.onnx" for i in range(n_parts)]

    for i, part in enumerate(parts):
        print(f"Part: {i+1}")

        first_node_in_part_info = node_info[part[0][0]]
        input_names = [name for name in first_node_in_part_info[0] ]

        part_outputs = set()
        for p in part:
            node_name = p[0]
            part_outputs.update(node_info[node_name][1])

        # Check if any node has an input that is not an output of another node in the part
        for p in part:
            node_name = p[0]
            for input_name in node_info[node_name][0]:
                if input_name not in part_outputs:
                    if input_name not in input_names:
                        input_names.append(input_name)


        last_node_in_part_info = node_info[part[-1][0]]
        output_names = [name for name in last_node_in_part_info[1]]
        sub_model = extract_model(onnx_model, input_names, output_names,output_paths[i])
        model_parts.append(sub_model)
    
    for i, model_part in enumerate(model_parts):
        input_data = load_json_input(json_input)
        print(f"Running inference on part {i + 1}...")
        session = ort.InferenceSession(model_part.SerializeToString())
        
        input_names = [input.name for input in session.get_inputs()]
        
        if i == 0:
            itermediate_outputs[input_names[0]] = input_data

        assert all(name in itermediate_outputs for name in input_names), "Input data dictionary keys must match the model input names."
        infer_input = {}
        for name in input_names:
            infer_input[name] = itermediate_outputs[name]

        with open(f"test_parts/part{i+1}_input.json", 'w') as json_file:
            infer_input_serializable = convert_and_flatten_ndarray_to_list(copy.deepcopy(infer_input))

            json.dump(infer_input_serializable, json_file, indent=4)  # indent=4 for pretty-printing


        results = session.run(None, infer_input)
        
        print(f"Inference results for part {i + 1}:", results)
    return results
# Function to convert NumPy arrays to lists
def convert_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            d[key] = convert_ndarray_to_list(value)
    return d


# Function to convert and flatten NumPy arrays to lists
def convert_and_flatten_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.reshape([-1]).tolist()  # Flatten and convert to list
        elif isinstance(value, dict):
            d[key] = convert_and_flatten_ndarray_to_list(value)
    return d

def get_split_models_and_inputs(onnx_model,json_input,n_parts = 4):
    itermediate_outputs = get_intermediate_outputs(model,input)
    model = onnx.load(onnx_model)
    nodes = model.graph.node
    node_info = {}
    initializers = {init.name for init in model.graph.initializer}

    node_info = {}
    for node in nodes:
        node_inputs = [input for input in node.input if input not in initializers and 'Constant' not in input]
        node_outputs = [output for output in node.output if output not in initializers and 'Constant' not in output]
        if node_inputs and node_outputs:
            node_info[node.name] = (node_inputs, node_outputs)

    parts = divide_nodes_into_parts(node_info, n_parts)

    output = {}

    # output_paths = [f"test_parts/part{i+1}_model.onnx" for i in range(n_parts)]

    for i, part in enumerate(parts):
        first_node_in_part_info = node_info[part[0][0]]
        input_names = [name for name in first_node_in_part_info[0] ]

        part_outputs = set()
        for p in part:
            node_name = p[0]
            part_outputs.update(node_info[node_name][1])

        # Check if any node has an input that is not an output of another node in the part
        for p in part:
            node_name = p[0]
            for input_name in node_info[node_name][0]:
                if input_name not in part_outputs:
                    if input_name not in input_names:
                        input_names.append(input_name)


        last_node_in_part_info = node_info[part[-1][0]]
        output_names = [name for name in last_node_in_part_info[1]]
        sub_model = extract_model(onnx_model, input_names, output_names)
        
        session = ort.InferenceSession(sub_model.SerializeToString())
        
        input_names = [input.name for input in session.get_inputs()]
        
        if i == 0:
            itermediate_outputs[input_names[0]] = load_json_input(json_input)
        assert all(name in itermediate_outputs for name in input_names), "Input data dictionary keys must match the model input names."
        infer_input = {}
        for name in input_names:
            infer_input[name] = itermediate_outputs[name]
     
        output[i] = {
            "id": i,
            "model": sub_model,
            "input": infer_input
        }
    return output



if __name__ == "__main__":
    # model = 'examples/onnx/nanoGPT/network.onnx'
    # input = 'examples/onnx/nanoGPT/input.json'
    
    # input = 'test_outputs/inputs.json'
    # model = 'test_outputs/part_11_model.onnx'
    
    # model = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
    # input = 'examples/onnx/mobilenet/input.json'

    # model = 'examples/onnx/mnist_gan/network.onnx'
    # input = 'examples/onnx/mnist_gan/input.json'

    
    model = 'examples/onnx/residual_block/model.onnx'
    input = 'examples/onnx/residual_block/input_working.json'

    full_model_result = run_inference(model, input)
    print(f'full model result:{full_model_result}')
    #get the output tensor(s) of every node node in the model during inference
    intermediate_results = get_intermediate_outputs(model,input)

    split_model_output = run_inference_on_split_model(model,
                                                      input,
                                                      intermediate_results)
    print(f'split model result:{split_model_output}')

