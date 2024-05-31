import onnx
import onnxruntime as ort
import numpy as np
import json
from onnx import helper
from onnx import shape_inference, ModelProto
import torch
from typing import List

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
    input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is float32
    if input_data.size != 3 * 224 * 224:
        raise ValueError(f"Input data must be of size {3 * 224 * 224}, but got {input_data.size}")
    input_data = input_data.reshape(1, 3, 224, 224)
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


def split_onnx_model(onnx_model_path, num_splits =2):

    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)

    # Get the total number of layers in the model
    total_layers = len(model.graph.node)
    layers_per_split = total_layers // num_splits  # Calculate the number of layers per split

    # Split the model into parts
    split_models = []
    start_index = 0

    for i in range(num_splits):
        end_index = start_index + layers_per_split
        
        if i == num_splits - 1:  # Make sure the last split includes all remaining layers
            end_index = total_layers
        
        if i != 0:
            part_nodes = model.graph.node[start_index:end_index]
        else:
            end_index = end_index+1
            part_nodes = model.graph.node[start_index:end_index]
        
        # Get the output of the last layer of this part
        part_output_name = part_nodes[-1].output[0]

        # Create a new ONNX graph for this part
        part_graph = helper.make_graph(
            part_nodes,
            model.graph.name,
            model.graph.input if i == 0 else [helper.make_tensor_value_info(prev_part_output_name, onnx.TensorProto.FLOAT, None)],  # Input is the model's input for the first part, otherwise the output of the previous part
            [helper.make_tensor_value_info(part_output_name, onnx.TensorProto.FLOAT, None)],  # Output of the last layer of this part
            model.graph.initializer
        )

        # Create ONNX model for this part
        part_model = helper.make_model(part_graph)
        split_models.append(part_model)

        # Update for next split
        start_index = end_index
        prev_part_output_name = part_output_name
    
    return split_models


#     # Define the nodes that belong to the first part (e.g., first half of the model)
#     first_part_nodes = model.graph.node[:split_layer_index+1]
#     # Define the nodes that belong to the second part (e.g., second half of the model)
#     second_part_nodes = model.graph.node[split_layer_index+1:]
    
#     # Get the output of the last layer of the first part
#     first_part_output_name = first_part_nodes[-1].output[0]

#     # Create a new ONNX graph for the first part
#     first_part_graph = helper.make_graph(
#         first_part_nodes,
#         model.graph.name,
#         model.graph.input,
#         [helper.make_tensor_value_info(first_part_output_name, onnx.TensorProto.FLOAT, None)],  # Output of the last layer of the first part
#         model.graph.initializer
#     )

#     second_part_input_name = first_part_output_name

#    # Create a new ONNX graph for the second part
#     second_part_graph = helper.make_graph(
#         second_part_nodes,
#         model.graph.name,
#         [helper.make_tensor_value_info(second_part_input_name, onnx.TensorProto.FLOAT, None)],  # Input is the output of the first part
#         model.graph.output,
#         model.graph.initializer
#     )
    
#     # Create ONNX models for the first and second parts
#     first_part_model:ModelProto = helper.make_model(first_part_graph)
#     second_part_model = helper.make_model(second_part_graph)

#     return first_part_model, second_part_model


def run_inference_on_split_model(model_parts: List[onnx.ModelProto], input_path: str):

    # Load and preprocess the JSON input
    input_data = load_json_input(input_path)

    # Run inference sequentially on each part
    input_for_next_part = input_data
    for i, model_part in enumerate(model_parts):
        print(f"Running inference on part {i + 1}...")
        results = run_inference(model_part.SerializeToString(), input_for_next_part)
        print(f"Inference results for part {i + 1}:", results)
        
        # Prepare the output of this part as the input for the next part
        input_for_next_part = results[0]  # Assuming the output is the first element of results

    # The final results after running inference on all parts
    final_results = input_for_next_part
    print("Final inference results:", final_results)
    return final_results

# def run_inference_on_split_model(model_part1:ModelProto,model_part2:ModelProto, input_path):

#     # Load and preprocess the JSON input
#     input_data = load_json_input(input_path)
#      # Run inference on first part
#     results_part1 = run_inference(model_part1.SerializeToString(), input_data)
#     print("Inference results for part 1:", results_part1)

#     # Extract output of first part as input for second part
#     input_data_part2 = results_part1[0]  # Assuming the output is the first element of results

#     # Run inference on second part
#     results_part2 = run_inference(model_part2.SerializeToString(), input_data_part2)

#     # Print final results
#     print("Inference results for part 2:", results_part2)

def extract_halfway_model(input_path, output_path):
    # Load the original ONNX model
    original_model = onnx.load(input_path)

    # Get the total number of layers in the model
    total_layers = len(original_model.graph.node)

     # Extract the input and output node names for the sub-model
    input_names = original_model.graph.node[0].input
    output_names = original_model.graph.node[-1].output

    # Calculate the index of the middle layer
    halfway_layer_index = total_layers // 2

    # Extract the sub-model with approximately half the layers
    onnx.utils.extract_model(input_path,output_path,input_names, output_names, True)


def main():
    print("ONNX version:", onnx.__version__)

    original_model_path = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
    original_input_path = 'examples/onnx/mobilenet/input.json'   
    output_path1 = 'from_onnx/mobilenet/model_part1.onnx'
    output_path2 = 'from_onnx/mobilenet/model_part2.onnx'

    split_models = split_onnx_model(original_model_path, num_splits=3)
    # onnx.save(split_models[0],output_path1)
    # onnx.save(split_models[1],output_path2)


    run_inference_on_split_model(split_models, original_input_path)
    
    run_inference_on_full_model(original_model_path,original_input_path)
    # run_inference_on_full_model()



if __name__ == "__main__":
    main()