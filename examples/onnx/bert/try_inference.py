import json
import numpy as np
import onnxruntime as ort

# -------------------------------
# 1. Load ONNX model
# -------------------------------
onnx_path = "examples/onnx/bert/bert_tiny.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Inspect model inputs / outputs
print("Model inputs:", session.get_inputs())
print("Model outputs:", session.get_outputs())

# -------------------------------
# 2. Load input JSON
# -------------------------------
with open("examples/onnx/bert/bert_tiny_input.json", "r") as f:
    data = json.load(f)

# Convert JSON back to numpy (float32 or int64 as required by your ONNX graph)
flat_array = np.array(data["input_data"][0], dtype=np.int64)  # int64 matches tokenizer IDs
flat_array = flat_array.reshape(1, -1)  # restore shape [1, 3*seq_len]

# -------------------------------
# 3. Run inference
# -------------------------------
outputs = session.run(
    None,   # request all outputs
    {"input": flat_array}
)

start_logits, end_logits = outputs
print("Start logits:", start_logits.shape)
print("End logits:", end_logits.shape)
