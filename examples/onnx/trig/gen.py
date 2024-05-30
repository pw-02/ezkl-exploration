import io
import numpy as np
from torch import nn
import torch.onnx
import torch
import torch.nn as nn
import torch.nn.init as init
import json


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(x)
        x = torch.cos(x)
        x = torch.sin(x)
        x = torch.tan(x)
        x = torch.acos(x)
        x = torch.asin(x)
        x = torch.atan(x)
        # x = torch.cosh(x)
        # x = torch.sinh(x)
        x = torch.tanh(x)
        # x = torch.acosh(x)
        # x = torch.asinh(x)
        # x = torch.atanh(x)
        return (-x).abs().sign()


def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 3]
    x = 0.1*torch.rand(1, *shape, requires_grad=True)
    torch_out = torch_model(x)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      # model input (or a tuple for multiple inputs)
                      x,
                      # where to save the model (can be a file or file-like object)
                      "network.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes=[shape, shape, shape],
                input_data=[d],
                output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump(data, open("input.json", 'w'))


if __name__ == "__main__":
    main()
