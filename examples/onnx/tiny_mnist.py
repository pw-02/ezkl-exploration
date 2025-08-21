# profile_mnist_cnn_export.py
import torch, torch.nn as nn, torch.nn.functional as F
import json

class TinyMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16*7*7, 10)

    def forward(self, x):
        # Optional: keep profiling prints for one run
        x1 = self.conv1(x)
        print("conv1:", x1.min().item(), x1.max().item())
        x = F.relu(x1)

        x2 = F.max_pool2d(x, 2)
        print("after pool1:", x2.min().item(), x2.max().item())

        x3 = self.conv2(x2)
        print("conv2:", x3.min().item(), x3.max().item())
        x = F.relu(x3)

        x4 = F.max_pool2d(x, 2)
        print("after pool2:", x4.min().item(), x4.max().item())

        x = torch.flatten(x4, 1)
        x5 = self.fc(x)
        print("fc:", x5.min().item(), x5.max().item())

        return x5

# Instantiate and set eval mode
model = TinyMNIST().eval()

# Dummy input (batch=1, channels=1, 28x28)
dummy_input = torch.randn(1, 1, 28, 28)

# Run once to print ranges
out = model(dummy_input)
print("logits:", out)

# ---- Export ONNX with fixed shape ----
torch.onnx.export(
    model, dummy_input, "profile_mnist_fixed.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13
)
print("Exported profile_mnist_fixed.onnx")

# ---- Save valid JSON input ----
# Flatten to [784] inside a list (consistent with earlier working format)
flat_input = dummy_input.flatten().tolist()
with open("profile_mnist_input.json", "w") as f:
    json.dump({"input_data": [flat_input]}, f)

print("Exported profile_mnist_input.json")
