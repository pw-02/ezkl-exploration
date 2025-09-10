import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer

# Tiny BERT model (2 layers, hidden size 128)
model_name = "google/bert_uncased_L-2_H-128_A-2"

# Load pretrained model + tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name, attn_implementation="eager")
model.eval()

# Fixed MLPerf-style input: seq_len=384, batch_size=1
sequence_length = 384
dummy_input = tokenizer(
    ["What is MLPerf?"], ["MLPerf is a benchmark suite for ML models."],
    padding="max_length",
    truncation=True,
    max_length=sequence_length,
    return_tensors="pt"
)

# Prepare inputs for ONNX
inputs = (
    dummy_input["input_ids"],
    dummy_input["attention_mask"],
    dummy_input["token_type_ids"]
)

# Export to ONNX
onnx_path = "examples/onnx/bert/bert_tiny_squad.onnx"
torch.onnx.export(
    model,
    inputs,
    onnx_path,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["start_logits", "end_logits"],
    opset_version=11,
    do_constant_folding=True,  # optimize for inference
    dynamic_axes=None  # fixed shape [1,384]
)

print(f"Exported ONNX model: {onnx_path}")

# 🔥 Run model once to capture outputs
with torch.no_grad():
    outputs = model(**dummy_input)

# Convert everything to lists
input_dict = {
    "input_ids": [dummy_input["input_ids"].tolist()],
    "attention_mask": [dummy_input["attention_mask"].tolist()],
    "token_type_ids": [dummy_input["token_type_ids"].tolist()],
}

output_dict = {
    "start_logits": [outputs.start_logits.tolist()],
    "end_logits": [outputs.end_logits.tolist()],
}

# Save both to JSON
all_data = {"input_data": input_dict, "output_data": output_dict}

json_path = "examples/onnx/bert/bert_tiny_squad_data.json"
with open(json_path, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Exported inputs + outputs: {json_path}")
