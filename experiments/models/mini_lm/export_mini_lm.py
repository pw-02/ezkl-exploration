from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import os

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

EXPORT_DIR = "experiments/models/mini_lm"
ONNX_PATH = f"{EXPORT_DIR}/model.onnx"
QUANT_PATH = f"{EXPORT_DIR}/model_int8.onnx"

os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Exporting MiniLM to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    export=True
)

model.save_pretrained(EXPORT_DIR)
tokenizer.save_pretrained(EXPORT_DIR)

print(f"ONNX model saved to: {ONNX_PATH}")

print("Quantizing to INT8...")
quantize_dynamic(
    ONNX_PATH,
    QUANT_PATH,
    weight_type=QuantType.QInt8
)

print(f"Quantized model saved to: {QUANT_PATH}")

print("Testing inference...")

session = ort.InferenceSession(
    QUANT_PATH,
    providers=["CPUExecutionProvider"]
)

text = "zero knowledge machine learning"

inputs = tokenizer(
    text,
    return_tensors="np",
    padding=True,
    truncation=True,
)

outputs = session.run(None, dict(inputs))

print("Success!")
print("Output shape:", outputs[0].shape)