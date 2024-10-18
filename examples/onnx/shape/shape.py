import torch
import torch.nn as nn

class ShapeGatherModel(nn.Module):
    def __init__(self):
        super(ShapeGatherModel, self).__init__()
    
    def forward(self, x, index):
        shape = x.shape  # Get the shape of the input tensor
        shape_tensor = torch.tensor(shape, dtype=torch.int64)  # Convert shape to tensor
        gathered = shape_tensor[index]  # Gather the element at the specified index
        return gathered

# Create the model instance
model = ShapeGatherModel()

# Print the model to check
print(model)

import torch.onnx

# Define dummy inputs
dummy_input = torch.randn(2, 3, 4)  # Shape of the input tensor
dummy_index = torch.tensor([1])  # Index to gather from shape

# Export the model to ONNX format
onnx_file_path = 'shape_gather_model.onnx'
torch.onnx.export(
    model, 
    (dummy_input, dummy_index),  # Tuple of inputs
    onnx_file_path, 
    input_names=['input', 'index'],  # Input names
    output_names=['gather_out'],  # Output names
    opset_version=12  # ONNX opset version
)

print(f"Model exported to {onnx_file_path}")
