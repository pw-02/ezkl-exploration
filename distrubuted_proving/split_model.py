import onnx
import onnxruntime as ort
from onnx import shape_inference, ModelProto
import json
import numpy as np
import os
from onnx.utils import Extractor

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



def load_json_input(input_path, input_shape):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is float
    
    # Try to reshape the input data to match the expected shape
    try:
        input_data = input_data.reshape(input_shape)
    except ValueError as e:
        raise ValueError(f"Input data cannot be reshaped from shape {input_data.shape} to the expected shape {input_shape}: {e}")
 

    # Check if the input data matches the expected shape
    if list(input_data.shape) != input_shape:
        raise ValueError(f"Input data shape {input_data.shape} does not match the expected shape {input_shape}")
 
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
    input_data = load_json_input(json_input, input_shape)

    # Run inference
    results = session.run(None, {input.name: input_data})
    intermediate_inference_outputs = {}
    # Print intermediate results
    for name, result in zip(session.get_outputs(), results):
        intermediate_inference_outputs[name.name] = result
    return intermediate_inference_outputs

def run_inference_on_onnx_model(model_path, input_file):
    model = onnx.load(model_path)
    session = ort.InferenceSession(model.SerializeToString())
    input = session.get_inputs()[0]
    input_shape = input.shape
    input_type = input.type
    input_data = load_json_input(input_file, input_shape)
    results = session.run(None, {input.name: input_data})
    return results



def split_onnx_model_at_every_node(onnx_model_path, json_input, itermediate_outputs, output_folder = 'tmp', save_to_file = False):

    models_with_inputs = []

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

    if save_to_file:
        os.makedirs(output_folder, exist_ok=True)

    for idx, node_name in enumerate(nodes):

        # model_path = f'{output_folder}/model.onnx'
        # input_path = f'{output_folder}/input.json'
        model_save_path = f'{output_folder}/split_{idx+1}_model.onnx'
        input_data_save_path = f'{output_folder}/split_{idx+1}_input.json'

        # print(f"Processing Split {idx+1}, Node Name: {node_name}")
        node_inputs, node_outputs = nodes[node_name]
        if save_to_file:
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
            input_data = load_json_input(json_input, input_shape)
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
        
        models_with_inputs.append((sub_model,proving_input))
    
    return models_with_inputs

if __name__ == "__main__":
    models_to_test = [
        # ('examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx', 'examples/onnx/mobilenet/input.json'),
        ('examples/onnx/mnist_gan/network.onnx', 'examples/onnx/mnist_gan/input.json')

    ]
    for onnx_file, input_file in models_to_test:

        full_model_result = run_inference_on_onnx_model(onnx_file, input_file)

        # Get the output tensor(s) of every node in the model during inference
        intermediate_results = get_intermediate_outputs(onnx_file, input_file)

        split_onnx_model_at_every_node(onnx_file, input_file,  intermediate_results, 'examples/split_models/mnist_gan')  

