import onnx
import numpy as np
import json

# Load the serialized TensorProto from the pb file
with open(r'C:\Users\pw\Downloads\tiny-yolov3-11.tar\tiny-yolov3-11\test_data_set_0\input_0.pb', 'rb') as f:
    tensor = onnx.TensorProto()
    tensor.ParseFromString(f.read())

# Convert TensorProto to numpy array
from onnx import numpy_helper
# Convert to numpy array
arr = numpy_helper.to_array(tensor)

# Prepare JSON structure
result = {
    "input_shapes": [list(arr.shape)],
    "input_data": [arr.flatten().tolist()]
}

# Write to JSON file
with open('input_0.json', 'w') as f:
    json.dump(result, f, indent=2)