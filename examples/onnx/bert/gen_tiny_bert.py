import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizer

import torch
from transformers import BertForQuestionAnswering

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        x is a single tensor shaped [batch_size, 3*seq_len]
        We'll split into input_ids, attention_mask, token_type_ids
        """
        batch_size, total_len = x.shape
        seq_len = total_len // 3

        # Split into 3 equal parts
        input_ids = x[:, :seq_len].long()
        attention_mask = x[:, seq_len:2*seq_len].long()
        token_type_ids = x[:, 2*seq_len:].long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.start_logits, outputs.end_logits



# -------------------------------
# 1. Load model + tokenizer
# -------------------------------
#
model_name = "google/bert_uncased_L-2_H-128_A-2"  # Tiny BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
model = BertWrapper(model)
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
# Concatenate into one input
flat_input = torch.cat(
    [
        dummy_input["input_ids"],
        dummy_input["attention_mask"],
        dummy_input["token_type_ids"]
    ],
    dim=1
)


# ------------------------------
# 3. Export to ONNX
# -------------------------------
onnx_path = "examples/onnx/bert/bert_tiny.onnx"
torch.onnx.export(
    model,
    flat_input,
    onnx_path,
    input_names=["input"],
    output_names=["start_logits", "end_logits"],
    opset_version=14,         # EZKL supports 9–18
    do_constant_folding=True,
    dynamic_axes=None         # EZKL needs fixed shapes
)
print(f"✅ Exported ONNX model: {onnx_path}")

# -------------------------------
# 4. Save inputs + outputs to JSON
# -------------------------------

input_list = flat_input.flatten().tolist()
with open("examples/onnx/bert/bert_tiny_input.json", "w") as f:
    json.dump({"input_data": [input_list]}, f, indent=2)

print("✅ Saved EZKL input JSON")

