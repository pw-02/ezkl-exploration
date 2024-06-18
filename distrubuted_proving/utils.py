import onnx
import onnxruntime as ort
import numpy as np
import json
from onnx import helper
from onnx import shape_inference, ModelProto
from typing import List
from onnx import ModelProto
from onnx_split import split_onnx, get_all_cut_points

def get_num_parameters(model = None):
    # Load the ONNX model
    if isinstance(model, str):
        model = onnx.load(model)

    # Initialize the parameter counter
    num_parameters = 0
    # Iterate through all the initializers (weights, biases, etc.)
    for initializer in model.graph.initializer:
        # Get the shape of the parameter
        param_shape = onnx.numpy_helper.to_array(initializer).shape
        # Calculate the total number of elements in this parameter
        param_size = np.prod(param_shape)
        # Add to the total parameter count
        num_parameters += param_size
    
    return num_parameters


def list_layers(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    layers = []
    for node in onnx_model.graph.node:
        layers.append(node.name)
    return layers

def load_onnx_model(model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model

def load_and_infer_model(model_path):
    model = onnx.load(model_path)
    inferred_model = shape_inference.infer_shapes(model)
    return inferred_model

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

def run_inference(model_path, input_data):
    # Initialize ONNX Runtime session
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    results = session.run(None, {input_name: input_data})
    return results


def run_inference_on_full_model(model_path, input_path):
    # Load and check the ONNX model
    onnx_model:ModelProto = load_onnx_model(model_path)

    # Load and preprocess the JSON input
    input_data = load_json_input(input_path)
    # Run inference
    results = run_inference(onnx_model.SerializeToString(), input_data)
    # Print results
    print("Inference results:", results)

def split_onnx_model(onnx_model_path, n_parts, max_parameters_threshold=np.inf, verbose=0, stats=False):

    if isinstance(onnx_model_path,str):
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = shape_inference.infer_shapes(onnx_model)
    else:
        onnx_model = onnx_model_path
        # onnx_model = shape_inference.infer_shapes(onnx_model)

    parts = split_onnx(onnx_model, n_parts=n_parts, cut_points=None, verbose=verbose, stats=stats, fLOG=None)
    updated_parts = []
    for idx, part_model in enumerate(parts):
        if get_num_parameters(model=part_model) > max_parameters_threshold and idx > 0:
            # Recursively split the part
            sub_parts = split_onnx_model(part_model, 2, max_parameters_threshold, verbose=verbose, stats=stats)
            updated_parts.extend(sub_parts)
        else:
            updated_parts.append(part_model)
    return updated_parts




def get_model_splits_inputs(model_parts: List[onnx.ModelProto], input_path: str):
    # Load and preprocess the JSON input
    input_data = load_json_input(input_path)
    inputs = []
    # Run inference sequentially on each part
    input_for_next_part = input_data
    for i, model_part in enumerate(model_parts):
        inputs.append(input_for_next_part)
        results = run_inference(model_part.SerializeToString(), input_for_next_part)
        input_for_next_part = results[0]  # Assuming the output is the first element of results
    return inputs


def split_onnx_mode_oldl(onnx_model_path, num_splits =2):

    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)

    if num_splits <=1:
        return [model]

    # Get the total number of layers in the model
    total_nodes = len(model.graph.node)
    nodes_per_split = total_nodes // num_splits  # Calculate the number of layers per split

    # Split the model into parts
    split_models = []
    start_index = 0
    print(f'initial_parameter_count: {get_num_parameters(model=model)}')  # Count parameters before splitting

    for i in range(num_splits):
        end_index = min(start_index + nodes_per_split, total_nodes)  # Ensure end index does not exceed total nodes
        
        part_nodes = model.graph.node[start_index:end_index]
        next_nodes = model.graph.node[end_index:end_index+nodes_per_split]

        # Collect initializers that are used as inputs in the next split
        split_input_names = [inp for node in part_nodes for inp in node.input]
        part_initializers = [init for init in model.graph.initializer if init.name in split_input_names]

        # Get the output name of the last node of this part
        part_output_name = part_nodes[-1].output[0] if part_nodes else None

        # Create a new ONNX graph for this part
        part_graph = helper.make_graph(
            part_nodes,
            model.graph.name,
            model.graph.input if i == 0 else [helper.make_tensor_value_info(prev_part_output_name, onnx.TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info(part_output_name, onnx.TensorProto.FLOAT, None)] if part_output_name else [],
            part_initializers
        )

         # Create ONNX model for this part
        part_model = helper.make_model(part_graph)
        split_models.append(part_model)

        #  # Update for next split
        start_index = end_index
        prev_part_output_name = part_output_name
        print(f'split_{i}_parameter_count: {get_num_parameters(model=part_model)}')
    
    return split_models



def run_inference_on_split_model(model_parts: List[onnx.ModelProto], input_path: str):

    # Load and preprocess the JSON input
    input_data = load_json_input(input_path)

    # Run inference sequentially on each part
    input_for_next_part = input_data
    for i, model_part in enumerate(model_parts):
        print(f"Running inference on part {i + 1}...")
        results = run_inference(model_part.SerializeToString(), input_for_next_part)
        #print(f"Inference results for part {i + 1}:", results)
        
        # Prepare the output of this part as the input for the next part
        input_for_next_part = results[0]  # Assuming the output is the first element of results

    # The final results after running inference on all parts
    final_results = input_for_next_part
    #print("Final inference results:", final_results)
    return final_results


def get_all_possible_cut_points(onnx_model_path, verbose=0, stats=False):
    if isinstance(onnx_model_path,str):
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = shape_inference.infer_shapes(onnx_model)
    else:
        onnx_model = onnx_model_path
        # onnx_model = shape_inference.infer_shapes(onnx_model)
    
    cutting_points = get_all_cut_points(onnx_model, verbose=verbose)
    return cutting_points


def split_at_provided_cut_points(onnx_model_path, cut_points_provided, verbose=0, stats=False):
    if isinstance(onnx_model_path,str):
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = shape_inference.infer_shapes(onnx_model)
    else:
        onnx_model = onnx_model_path
        # onnx_model = shape_inference.infer_shapes(onnx_model)
    
    parts = split_onnx(onnx_model, n_parts=None, cut_points=cut_points_provided, verbose=verbose, stats=stats, fLOG=None)
    return parts

def save_parts_as_onnx_files(original_model_path, model_parts: List[onnx.ModelProto]):
    for i, model_part in enumerate(model_parts):
        part_path = f"{original_model_path}_part_{i+1}.onnx"
        print(f"saving: {part_path}...")
        part_binary_string = model_part.SerializeToString()
        with open(part_path, "wb") as file:
            file.write(part_binary_string)

def main():
    print("ONNX version:", onnx.__version__)

    original_model_path = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
    original_input_path = 'examples/onnx/mobilenet/input.json'   
    
    # original_model_path = 'examples/onnx/random_forest/network.onnx'
    # original_input_path = 'examples/onnx/random_forest/input.json'

    # original_model_path = 'examples/onnx/mnist_classifier/network.onnx'
    # original_input_path = 'examples/onnx/mnist_classifier/input.json'

    # original_model_path = 'examples/onnx/mnist_gan/network.onnx'
    # original_input_path = 'examples/onnx/mnist_gan/input.json'

    # original_model_path = 'examples/onnx/little_transformer/network.onnx'
    # original_input_path = 'examples/onnx/little_transformer/input.json'   

    # original_model_path = 'examples/onnx/efficient_net/efficientnet-lite4-11.onnx'
    # original_input_path = 'examples/onnx/shuffle_net/input.json'

    # original_model_path = 'examples/onnx/nanoGPT/gptneox_Opset16.onnx'
    # original_input_path = 'examples/onnx/nanoGPT/input.json'  

    #split_models = split_onnx_model(original_model_path, n_parts=4, max_parameters_threshold=np.inf, verbose=20, stats=False)

    cut_points = get_all_possible_cut_points(original_model_path, verbose=0, stats=False)
    print(f"All possible split points({len(cut_points)}): {cut_points}\n")

    verbose = 1
    split_models = split_at_provided_cut_points(original_model_path, cut_points_provided=cut_points, verbose=verbose, stats=False)
    # split_models = split_onnx_model(original_model_path, num_splits=2)
    
    run_inference_on_split_model(split_models, original_input_path)
    run_inference_on_full_model(original_model_path,original_input_path)

    save_parts_as_onnx_files(original_model_path, split_models)
    
    
    # run_inference_on_full_model()



if __name__ == "__main__":
    main()