import random
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import json



input_size = 512   # e.g., sequence features
hidden_size = 1024 # larger hidden state
num_layers = 1     # stacked LSTMs
seq_len = 10


model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=False)
# x = torch.randn(1, 3)  # make a sequence of length 5
x = torch.randn(seq_len, 1, input_size)  # (seq_len, batch, input_size)

# Flips the neural net into inference mode
model.eval()
model.to('cpu')

# Export the model
torch.onnx.export(model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  r"examples\onnx\lstm\lstm_network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data_json = dict(input_data=[data_array])

print(data_json)

# Serialize data into file:
json.dump(data_json, open( r"examples\onnx\lstm\lstm_input.json", 'w'))
