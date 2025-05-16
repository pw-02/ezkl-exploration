import json
import numpy as np
import onnxruntime as ort


def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_model_input(input_data_path, expected_shape, input_type, idx=0):
    """Format input tensor from JSON to match ONNX model expectations."""
    expected_shape = [-1 if dim == 'batch_size' else dim for dim in expected_shape]
    input_data = load_json_input(input_data_path)['input_data']

    if input_type == 'tensor(float)':
        reshaped_input = np.array(input_data, dtype=np.float32).reshape(expected_shape)
    elif input_type == 'tensor(int64)':
        reshaped_input = np.array(input_data[idx] if expected_shape else input_data[0][0], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    if 'gpt' in str(input_data_path).lower():
        reshaped_input = np.reshape(input_data, (1, 64))

    return reshaped_input


def run_model_inference(onnx_model_path, input_data_path):
    """Run inference with ONNX Runtime using formatted input."""
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    input_tensor = format_model_input(input_data_path, input_shape, input_type)
    outputs = session.run(None, {input_name: input_tensor})
    return outputs


    