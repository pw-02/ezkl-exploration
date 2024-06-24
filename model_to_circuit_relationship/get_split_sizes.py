import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import ezkl
import csv
from collections import Counter
from example_models import example_models_list, GPT
# Function to get all operations in the forward pass of a PyTorch model
def get_pytorch_forward_operations(model, input_data):
    operations = []
    layer_types = Counter()

    def forward_hook(module, input, output):
        operations.append(type(module).__name__)
        layer_types[type(module).__name__] += 1

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)
    # input_tensor = torch.from_numpy(input_data)
    model(input_data)
    # model(input_data)

    for hook in hooks:
        hook.remove()
    # return operations, dict(layer_types)
    return dict(layer_types)


# Function to count the number of parameters in a PyTorch model
def count_pytorch_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params

# Function to count and list different types of layers in a PyTorch model
def get_pytorch_model_layer_types(model):
    layer_types = Counter()
    for layer in model.modules():
        layer_types[type(layer).__name__] += 1
    return dict(layer_types)

def get_onnx_forward_operations(onnx_model_path):
    model = onnx.load(onnx_model_path)
    layer_types = Counter(node.op_type for node in model.graph.node)
    return dict(layer_types)

# Function to count model parameters in an ONNX model
def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
    model_parameters = sum(np.prod(initializer.dims) for initializer in model.graph.initializer)
    return model_parameters

# Save input to JSON
def save_input_to_json(input_data, filename):
    with open(filename, 'w') as f:
        json.dump(input_data.tolist(), f)

# Load input from JSON
def load_input_from_json(filename):
    with open(filename, 'r') as f:
        input_list = json.load(f)
    return np.array(input_list, dtype=np.float32)

# Run inference on ONNX model and get output
def run_onnx_model_inference(onnx_model_path, input_data):

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    results = session.run(None, {input_name: input_data})
    return results

# Run inference on PyTorch model and get output
def run_pytorch_model_inference(model):
    
    with torch.no_grad():
            input_tensor =model.dummy_input()
            output = model(input_tensor).numpy()
    return input_tensor, output

def compare_outputs(pytorch_output, onnx_output):
    return np.allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05)

def compute_ratio(numerator, denominator):
    if denominator == 0:
        return None
    ratio = numerator/ denominator
    return ratio

if __name__ == "__main__":
    import os
    import glob
    # List of different model configurations to test
    # Iterate through each model configuration
    onnx_files = glob.glob(os.path.join("model_to_circuit_relationship/mobilenet_split", '**', '*.onnx'), recursive=True)


    for idx, onnx_model_path in enumerate(onnx_files):

        total_params_onnx = count_onnx_model_parameters(onnx_model_path)
        print(f"Total number of parameters in the ONNX model: {total_params_onnx}")
        
        # Generate EZKL settings
        settings_path = f"settings_{idx}.json"
        ezkl.gen_settings(onnx_model_path, settings_path)

        # Load EZKL settings
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)

        # Create output dictionary
        data_dict = {
            "model_name":onnx_model_path,
            "num_model_params(onnx)": total_params_onnx,
            "logrows_in_circuit": settings_data["run_args"]["logrows"],
            "rows_in_circuit": settings_data["num_rows"],
            "assignments_circuit": settings_data["total_assignments"],
            "circuit_rows/onnx_model_param": compute_ratio(settings_data["num_rows"], total_params_onnx),
            "circuit_assignments/onnx_model_param": compute_ratio(settings_data["total_assignments"], total_params_onnx),
        }

        csv_file = 'model_to_circuit_relationship/split_circuit_comparison.csv'
        file_exists = os.path.isfile(csv_file)

        # Print the output dictionary
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if the file is new
            writer.writerow(data_dict)  # Write data as a new row
