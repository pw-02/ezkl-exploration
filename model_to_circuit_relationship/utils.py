import onnx
import numpy as np
import onnxruntime as ort

# Function to count the number of parameters in a PyTorch model
def count_pytorch_model_parameters(model, trainable_params_only  = False):
    if trainable_params_only:
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        
    return total_params

# Function to count model parameters in an ONNX model
def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
    model_parameters = sum(np.prod(initializer.dims) for initializer in model.graph.initializer)
    return model_parameters

# Run inference on ONNX model and get output
def run_onnx_model_inference(onnx_model_path, input_data):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    results = session.run(None, {input_name: input_data})
    return results
