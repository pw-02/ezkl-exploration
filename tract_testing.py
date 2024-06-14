import tract
import numpy as np


# Load the ONNX model
model_path = "examples/onnx/1l_average/network.onnx"
model = tract.onnx().model_for_path(model_path)
print(f"Model: {model}")

# Optimize the model
model = model.into_optimized()

# Optionally, make the model runnable
runnable = model.into_runnable()

# Prepare some dummy input data (this should match the model's expected input shape)
# For example, if the model expects a 1D array of length 10:
dummy_input = np.random.rand(10).astype(np.float32)

# Run inference
result = runnable.run([dummy_input])

# Print the result
print("Inference result:", result)
pass