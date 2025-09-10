import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer

# -------------------------------
# 1. Load model + tokenizer
# -------------------------------
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
onnx_path = "examples/onnx/bert/bert_tiny_squad.onnx"
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
with torch.no_grad():
    outputs = model(**dummy_input)

input_dict = {
    "input_ids": dummy_input["input_ids"].tolist(),
    "attention_mask": dummy_input["attention_mask"].tolist(),
    "token_type_ids": dummy_input["token_type_ids"].tolist(),
}

output_dict = {
    "start_logits": outputs.start_logits.tolist(),
    "end_logits": outputs.end_logits.tolist()
}

all_data = {"input_data": input_dict, "outputs": output_dict}

json_path = "examples/onnx/bert/bert_tiny_squad_data.json"
with open(json_path, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"✅ Exported test vectors: {json_path}")
