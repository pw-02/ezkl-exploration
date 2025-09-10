import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer

# -------------------------------
# 1. Load model + tokenizer
# -------------------------------
#
model_name = "google/bert_uncased_L-2_H-128_A-2"  # Tiny BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
model.eval()

# -------------------------------
# 2. Dummy input
# -------------------------------
sequence_length = 32  # keep small for EZKL
dummy_input = tokenizer(
    ["What is MLPerf?"], 
    ["MLPerf is a benchmark suite for ML models."],
    padding="max_length",
    truncation=True,
    max_length=sequence_length,
    return_tensors="pt"
)

inputs = (
    dummy_input["input_ids"],
    dummy_input["attention_mask"],
    dummy_input["token_type_ids"]
)

# -------------------------------
# 3. Export to ONNX
# -------------------------------
onnx_path = "examples/onnx/bert/bert_tiny.onnx"
torch.onnx.export(
    model,
    inputs,
    onnx_path,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["start_logits", "end_logits"],
    opset_version=14,         # EZKL supports 9–18
    do_constant_folding=True,
    dynamic_axes=None         # EZKL needs fixed shapes
)
print(f"✅ Exported ONNX model: {onnx_path}")

# -------------------------------
# 4. Save inputs + outputs to JSON
# -------------------------------

input_list = [
    dummy_input["input_ids"].flatten().tolist(),
    dummy_input["attention_mask"].flatten().tolist(),
    dummy_input["token_type_ids"].flatten().tolist()
]

with open("bert_tiny_input.json", "w") as f:
    json.dump({"input_data": input_list}, f, indent=2)

