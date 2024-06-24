import onnx
import onnxruntime as ort
from onnx import helper
import json
import numpy as np
from typing import List
from onnx import shape_inference, ModelProto
from onnx.utils import Extractor
import os

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

    # Collect all node outputs
    for node in nodes:
        all_node_outputs.extend([output for output in node.output])

    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    
    # Add value_info of intermediate nodes to model's outputs
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in all_node_outputs:
            value_info_protos.append(helper.make_tensor_value_info(node.name, node.type.tensor_type.elem_type, 
                                                                   [dim.dim_value for dim in node.type.tensor_type.shape.dim]))
    model.graph.output.extend(value_info_protos)
    
    # Load and preprocess the JSON input
    input_data = load_json_input(json_input)

    # Run inference
    session = ort.InferenceSession(model.SerializeToString())
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
    check_model: bool = True,
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



def run_inference_on_split_model(onnx_model, json_input, n_parts, itermediate_outputs):
    model = onnx.load(onnx_model)
    nodes = model.graph.node
    node_info = {}
    for node in nodes:
        node_inputs = [input for input in node.input]
        node_outputs = [output for output in node.output]
        if node_inputs and node_outputs:
            node_info[node.name] = (node_inputs, node_outputs)
    
    def divide_nodes_into_parts(node_info, parts_count):
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
        # Print the nodes in each part
        for i, part in enumerate(parts):
            print(f"Part {i+1}:")
            for info in part:
                print(f"Node Name: {info[0]}, Input: {info[1]}, Output: {info[2]}")
            print()
        return parts

    # def extract_sub_model(input_path, output_path, input_names, output_names):  
    #         onnx.utils.extract_model(input_path, output_path, input_names, output_names)

    parts = divide_nodes_into_parts(node_info, n_parts)

    output_paths = [f"test_parts/part{i+1}_model.onnx" for i in range(n_parts)]
    model_parts = []

    for i, part in enumerate(parts):
        print(f"Part: {i+1}")

        first_node_in_part_info = node_info[part[0][0]]
        input_names = [name for name in first_node_in_part_info[0]  if not 'Constant' in name and not 'classifier' in name and not name.startswith('onnx::') and not name.startswith('classifier')]
        
        part_outputs = set()
        for p in part:
            node_name = p[0]
            part_outputs.update(node_info[node_name][1])

        # Check if any node has an input that is not an output of another node in the part
        for p in part:
            node_name = p[0]
            for input_name in node_info[node_name][0]:
                if input_name not in part_outputs and not input_name.startswith('onnx::') and not 'Constant' in input_name and not 'classifier' in input_name:
                    if input_name not in input_names:
                        input_names.append(input_name)


        last_node_in_part_info = node_info[part[-1][0]]
        output_names = [name for name in last_node_in_part_info[1] if not name.startswith('onnx::') and not name.startswith('onnx::')]
        sub_model = extract_model(onnx_model, input_names, output_names)
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

        results = session.run(None, infer_input)
        if len(results) > 1:
            pass
        
        print(f"Inference results for part {i + 1}:", results)
    return results


if __name__ == "__main__":
    onnx_model = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
    json_input = 'examples/onnx/mobilenet/input.json'
    full_model_result = run_inference(onnx_model, json_input)
    #get the output tensor(s) of every node node in the model during inference
    intermediate_results = get_intermediate_outputs(onnx_model,json_input)

    split_model_output = run_inference_on_split_model(onnx_model,
                                                      json_input,
                                                      65,
                                                      intermediate_results)
    
