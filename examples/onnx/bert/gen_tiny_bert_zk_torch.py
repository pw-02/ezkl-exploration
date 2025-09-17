import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# MLPerf uses BERT-Large (uncased, 24-layer, hidden size 1024)
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load pretrained model + tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name,attn_implementation="eager")
model.eval()

# Fixed MLPerf benchmark input: seq_len=384, batch_size=1
sequence_length = 384
dummy_input = tokenizer(
    ["What is MLPerf?"], ["MLPerf is a benchmark suite for ML models."],
    padding="max_length",
    truncation=True,
    max_length=sequence_length,
    return_tensors="pt"
)

# Prepare inputs for ONNX
inputs = (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"])

# Export to ONNX (opset >= 13 is fine, 14+ for newer ops)
torch.onnx.export(
    model,
    inputs,
    "buildmodels/bert/bert_large_squad.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["start_logits", "end_logits"],
    opset_version=11,
    dynamic_axes=None   # 🚫 no dynamic axes (fixed shape 1x384)
)

print("Exported ONNX model: bert_large_squad.onnx")
