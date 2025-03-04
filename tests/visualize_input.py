import numpy as np
import matplotlib.pyplot as plt
import json
# Assuming input_data is a 1D list with 784 values (28x28)

def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)  # Expecting a nested list from np.array().tolist()

def format_model_input(input_data_path, expected_shape, input_type, idx = 0):
    expected_shape = [-1 if dim == 'batch_size' else dim for dim in expected_shape]

    input_data = load_json_input(input_data_path)['input_data']
    input_data = np.array(input_data, dtype=np.float32)  # Convert back to NumPy array
    
    reshaped_input = input_data.reshape(expected_shape)
    return reshaped_input



