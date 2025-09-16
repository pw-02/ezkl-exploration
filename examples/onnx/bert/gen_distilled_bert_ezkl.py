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
# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip
model_name = "google/bert_uncased_L-4_H-256_A-4"  # Tiny BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name, attn_implementation="eager")

sequence_length = 96  # keep small for EZKL
#seed seed for reproducibility
torch.manual_seed(42)
input_ids = torch.randint(0, 100, (1, sequence_length))      # fake vocab IDs
attention_mask = torch.ones((1, sequence_length), dtype=torch.int64)
token_type_ids = torch.zeros((1, sequence_length), dtype=torch.int64)
# Concatenate into one input
# save_input_path = "examples/onnx/bert/bert_tiny_input.json"
# save_onnx_path = "examples/onnx/bert/bert_tiny.onnx"
save_input_path = "examples/onnx/bert/bert_uncased_L-4_H-256_A-4_input.json"
save_onnx_path = "examples/onnx/bert/bert_uncased_L-4_H-256_A-4.onnx"
if use_wrapper := False: #set to True to for working with ezkl

    inputs = torch.cat([input_ids,attention_mask,token_type_ids],dim=1)
    model = BertWrapper(model)
    input_list = inputs.flatten().tolist()
    with open(save_input_path, "w") as f:
        json.dump({"input_data": [input_list]}, f, indent=2)
    model.eval()
    # ------------------------------
    # 3. Export to ONNX
    # -------------------------------
    torch.onnx.export(
        model,
        inputs,
        save_onnx_path,
        export_params=True,        # store the trained parameter weights inside the model file
        input_names=["input"],
        output_names=["start_logits", "end_logits"],
        opset_version=11,         # EZKL supports 9–18
        do_constant_folding=True,
        # dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
        #               'output': {0: 'batch_size'}})
        dynamic_axes=None)

else:

    inputs = (input_ids, attention_mask, token_type_ids)
    input_list = [input_ids.flatten().tolist(), attention_mask.flatten().tolist(), token_type_ids.flatten().tolist()]
    with open(save_input_path, "w") as f:
        json.dump({"input_data": input_list}, f, indent=2)

    model.eval()
    # ------------------------------
    # 3. Export to ONNX
    # -------------------------------
    torch.onnx.export(
        model,
        inputs,
        save_onnx_path,
        export_params=True,        # store the trained parameter weights inside the model file
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["start_logits", "end_logits"],
        opset_version=11,         # EZKL supports 9–18
        do_constant_folding=True,
        # dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
        #               'output': {0: 'batch_size'}})
        dynamic_axes=None)

print(f"✅ Exported ONNX model: {save_onnx_path}")

# -------------------------------
# 4. Save inputs + outputs to JSON
# -------------------------------