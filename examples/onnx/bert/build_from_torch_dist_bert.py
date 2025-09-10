import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

# -------------------------------
# 1. Load model + tokenizer
# -------------------------------
model_name = "distilbert-base-uncased"  # Tiny BERT
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

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
    dummy_input["attention_mask"]
)

# -------------------------------
# 3. Export to ONNX
# -------------------------------
onnx_path = "examples/onnx/bert/distilbert_squad.onnx"
torch.onnx.export(
    model,
    inputs,
    onnx_path,
    input_names=["input_ids", "attention_mask"],
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
    dummy_input["attention_mask"].flatten().tolist()
]

with open("examples/onnx/bert/distilbert_input.json", "w") as f:
    json.dump({"input_data": input_list}, f, indent=2)
