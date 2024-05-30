import torch
from torch import nn
from ezkl import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return x[:,1:2]


circuit = MyModel()
export(circuit, input_shape=[3, 2])
