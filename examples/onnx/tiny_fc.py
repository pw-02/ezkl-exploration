# tiny_fc_export.py
import torch, torch.nn as nn, torch.nn.functional as F
import json

# ----- Define a very small model -----
class TinyFC(nn.Module):
    def __init__(self):
        super().__init__()
        # Two-layer perceptron: 784 -> 16 -> 10
        self.fc1 = nn.Linear(28*28, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)   # flatten [1,1,28,28] -> [1,784]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----- Instantiate model -----
model = TinyFC().eval()

# Dummy input (batch=1, channels=1, 28x28)
dummy_input = torch.randn(1, 1, 28, 28)

# ----- Export ONNX with fixed shape -----
torch.onnx.export(
    model,
    dummy_input,
    "tiny_fc.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=13
)
print("Exported tiny_fc.onnx")

# ----- Save valid JSON input -----
# Flatten the tensor: [1,1,28,28] -> [784]
flat_input = dummy_input.flatten().tolist()
with open("tiny_fc_input.json", "w") as f:
    json.dump({"input_data": [flat_input]}, f)

print("Exported tiny_fc_input.json")
