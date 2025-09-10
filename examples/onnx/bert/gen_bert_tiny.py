import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer

# -------------------------------
# 1. Load model + tokenizer
# -------------------------------
model_name = "prajjwal1/bert-tiny"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name, attn_implementation="eager")
model.eval()

# -------------------------------
# 2. Dummy input
# -------------------------------
sequence_length = 32  # keep small for EZKL circuits
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
    dummy_input["token_type_ids"],
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
    opset_version=11,         # EZKL supports 9–18
    do_constant_folding=True,
    dynamic_axes=None         # fixed shapes for zk
)
print(f"✅ Exported ONNX model: {onnx_path}")

# -------------------------------
# 4. Save inputs to JSON (EZKL format)
# -------------------------------
input_list = [
    dummy_input["input_ids"].flatten().tolist(),
    dummy_input["attention_mask"].flatten().tolist(),
    dummy_input["token_type_ids"].flatten().tolist()
]

with open("examples/onnx/bert/bert_tiny_input.json", "w") as f:
    json.dump({"input_data": input_list}, f, indent=2)

print("✅ Saved input data: examples/onnx/bert/bert_tiny_input.json")

# # -------------------------------
# # 5. (Optional) Save reference outputs
# # -------------------------------
# with torch.no_grad():
#     outputs = model(**dummy_input)

# output_dict = {
#     "start_logits": outputs.start_logits.squeeze().tolist(),
#     "end_logits": outputs.end_logits.squeeze().tolist(),
# }

# with open("examples/onnx/bert/bert_output.json", "w") as f:
#     json.dump(output_dict, f, indent=2)

# print("✅ Saved reference outputs: examples/onnx/bert/bert_output.json")
