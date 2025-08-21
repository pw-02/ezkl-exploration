# tiny_mnist_cnn_export_fixed_flat.py
import torch, torch.nn as nn, torch.nn.functional as F
import json

class TinyMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16*7*7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Instantiate model
model = TinyMNIST().eval()

# Dummy input (batch=1, channels=1, 28x28)
dummy_input = torch.randn(1, 1, 28, 28)

# Export ONNX with *fixed* shape
torch.onnx.export(
    model, dummy_input, "tiny_mnist_fixed.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13
)
print("Exported tiny_mnist_fixed.onnx")

# Save valid JSON input in flat structure
# Shape = [1,1,28,28] -> flatten to [784] inside a list
flat_input = dummy_input.flatten().tolist()
with open("tiny_mnist_input.json", "w") as f:
    json.dump({"input_data": [flat_input]}, f)

print("Exported tiny_mnist_input.json (flat)")
