import onnx
import numpy as np
import onnxruntime as ort
from collections import Counter
from collections import Counter, defaultdict

# Function to count the number of parameters in a PyTorch model
def count_pytorch_model_parameters(model, trainable_params_only  = False):
    if trainable_params_only:
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        
    return total_params

def new_gelu(x):
    import torch
    import math
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

def get_onnx_model_operations(onnx_model_path, include_param_count=True):
    model = onnx.load(onnx_model_path)
    
    # Initialize counters
    layer_types = Counter()
    param_counts_by_layer_type = defaultdict(int)
    
    # Create a dictionary for initializers for quick lookup
    initializer_dict = {tensor.name: tensor for tensor in model.graph.initializer}
    
    for node in model.graph.node:
        layer_type = node.op_type
        layer_types[layer_type] += 1
        total_params = 0
        
        if include_param_count:
            for input_name in node.input:
                if input_name in initializer_dict:
                    tensor = initializer_dict[input_name]
                    # Calculate the number of parameters in this tensor
                    param_count = 1
                    for dim in tensor.dims:
                        param_count *= dim
                    total_params += param_count
            
            param_counts_by_layer_type[layer_type] += total_params
    
    # Combine the data into a single dictionary
    combined_data = {
        layer_type: {
            'count': layer_types[layer_type],
            'total_parameters': param_counts_by_layer_type[layer_type] if include_param_count else None
        }
        for layer_type in layer_types
    }
    
    return combined_data
    
def count_weights_and_tensors_in_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    
    total_weights = 0
    total_input_size = 0
    total_output_size = 0

    # Count weights in initializers
    for initializer in model.graph.initializer:
        total_weights += len(onnx.numpy_helper.to_array(initializer).flatten())

    # Count input tensor sizes
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        total_input_size += int(np.prod(shape))

    # Count output tensor sizes
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        total_output_size += int(np.prod(shape))

    return total_weights + total_input_size + total_output_size

def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
     # Get the list of initializers (weights and biases)
    initializers = model.graph.initializer
    total_weights = 0
    for initializer in initializers:
        # Convert the initializer to a numpy array to get its size
        array = onnx.numpy_helper.to_array(initializer)
        total_weights += np.prod(array.shape)
    return total_weights


import onnx
import numpy as np

def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
    
    # Initialize total_weights counter
    total_weights = 0
    
    # Count parameters in initializers (weights and biases)
    initializers = model.graph.initializer
    for initializer in initializers:
        array = onnx.numpy_helper.to_array(initializer)
        total_weights += np.prod(array.shape)
    
    # Count parameters in constants
    # Constants might be in the value_info field if they are not initializers
    for value_info in model.graph.value_info:
        if value_info.name in [init.name for init in initializers]:
            continue  # Skip if already counted in initializers
        # Look for any additional tensor constants in nodes (as some constants might not be in initializers)
        for node in model.graph.node:
            for input_name in node.input:
                if input_name == value_info.name:
                    # Placeholder for actual tensor data; not available here
                    # Add more detailed extraction if needed based on model specifics
                    pass
    return total_weights


def count_onnx_model_operations(onnx_model_path):
    model = onnx.load(onnx_model_path)
    nodes = model.graph.node
    num_operations = len(nodes)
    return num_operations



# Run inference on ONNX model and get output
def run_onnx_model_inference(onnx_model_path, input_data):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    results = session.run(None, {input_name: input_data})
    return results
