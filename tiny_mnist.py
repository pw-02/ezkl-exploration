# tiny_mnist_cnn_export.py
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

# Export ONNX
torch.onnx.export(
    model, dummy_input, "tiny_mnist.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
)

print("Exported tiny_mnist.onnx")

# Save valid JSON input
# Flatten to a nested list: [batch, channels, height, width]
input_list = dummy_input.tolist()
with open("tiny_mnist_input.json", "w") as f:
    json.dump({"input": input_list}, f)

print("Exported tiny_mnist_input.json")
