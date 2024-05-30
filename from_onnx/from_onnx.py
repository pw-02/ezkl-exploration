import onnx
import onnxruntime as ort
import numpy as np
import json
from onnx import helper
from onnx import shape_inference, ModelProto
from utils import list_layers
import torch

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
    input_data = np.array(input_data['input_data'], dtype=np.int64)
    # if input_data.size != 3 * 224 * 224:
    #     raise ValueError(f"Input data must be of size {3 * 224 * 224}, but got {input_data.size}")
    # input_data = input_data.reshape(1, 3, 224, 224)
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

def split_model_at_layer(original_model:ModelProto, split_layer):
    # Find the index of the split_layer node
    split_layer_index = None
    for i, node in enumerate(original_model.graph.node):
        # print(node.name)
        if node.name == split_layer:
            split_layer_index = i
            break
    # if split_layer_index is None:
    #     raise ValueError(f"Split layer '{split_layer}' not found in the model.")

    # Get the total number of layers in the model
    total_layers = len(original_model.graph.node)
    halfway_layer_index = total_layers // 2

    split_layer_index = halfway_layer_index
    # Define the nodes that belong to the first part (e.g., first half of the model)
    first_part_nodes = original_model.graph.node[:split_layer_index+1]
    # Define the nodes that belong to the second part (e.g., second half of the model)
    second_part_nodes = original_model.graph.node[split_layer_index+1:]
    
    # Get the output of the last layer of the first part
    first_part_output_name = first_part_nodes[-1].output[0]

    # Create a new ONNX graph for the first part
    first_part_graph = helper.make_graph(
        first_part_nodes,
        original_model.graph.name,
        original_model.graph.input,
        [helper.make_tensor_value_info(first_part_output_name, onnx.TensorProto.FLOAT, None)],  # Output of the last layer of the first part
        original_model.graph.initializer
    )

    second_part_input_name = first_part_output_name

   # Create a new ONNX graph for the second part
    second_part_graph = helper.make_graph(
        second_part_nodes,
        original_model.graph.name,
        [helper.make_tensor_value_info(second_part_input_name, onnx.TensorProto.FLOAT, None)],  # Input is the output of the first part
        original_model.graph.output,
        original_model.graph.initializer
    )
    
    # Create ONNX models for the first and second parts
    first_part_model:ModelProto = helper.make_model(first_part_graph)
    second_part_model = helper.make_model(second_part_graph)


    return first_part_model, second_part_model




def run_inference_on_split_model(model_part1:ModelProto,model_part2:ModelProto, input_path):

    # Load and preprocess the JSON input
    input_data = load_json_input(input_path)
     # Run inference on first part
    results_part1 = run_inference(model_part1.SerializeToString(), input_data)

    # Extract output of first part as input for second part
    input_data_part2 = results_part1[0]  # Assuming the output is the first element of results

    # Run inference on second part
    results_part2 = run_inference(model_part2.SerializeToString(), input_data_part2)

    # Print final results
    print("Inference results for part 1:", results_part1)
    print("Inference results for part 2:", results_part2)

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


    original_model_path = 'examples/onnx/nanoGPT/network.onnx'
    original_input_path = 'examples/onnx/nanoGPT/input.json'   
    output_path1 = 'from_onnx/model_part1.onnx'
    output_path2 = 'from_onnx/model_part2.onnx'
    print(len(list_layers(original_model_path)))

    run_inference_on_full_model(original_model_path,original_input_path)
    inferred_model = load_and_infer_model(original_model_path)
    # Split the model at a specific layer
    split_layer_name = "mobilenetv20_features_conv0_fwd"
    model_part1, model_part2 = split_model_at_layer(inferred_model, split_layer_name)
    onnx.save(model_part1,output_path1)
    onnx.save(model_part2,output_path2)

    run_inference_on_split_model(model_part1,model_part2, original_input_path)

    # run_inference_on_full_model()



if __name__ == "__main__":
    main()