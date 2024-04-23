# make sure you have the dependencies required here already installed
import ezkl
import os
import json
import time
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import Adam  # Import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# uncomment for more descriptive logging
FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

        # Fully connected layers / Dense block
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
        self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

    def forward(self, x):
        # Convolutional block
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
        return x

def normalize_img(image, label):
  return torch.round(image), label

def get_model(batch_size=64, max_epochs=1):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = batch_size
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LeNet().to(device)
    adam = Adam(model.parameters())  # Using Adam with a learning rate of 1e-3
    loss_fn = CrossEntropyLoss()
    all_epoch = max_epochs
    prev_acc = 0

    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            # normalize the image to 0 or 1 to reflect the inputs from the drawing board
            train_x = train_x.round()
            train_label = train_label.to(device)
            adam.zero_grad()  # Use adam optimizer
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            adam.step()  # Use adam optimizer
        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            # normalize the image to 0 or 1 to reflect the inputs from the drawing board
            test_x = test_x.round()
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('test accuracy: {:.3f}'.format(acc), flush=True)
    
    # # Fetch a single data point from the train_dataset
    # # Ensure train_dataset is already loaded and accessible
    train_data_point, _ = next(iter(train_dataset))
    train_data_point = train_data_point.unsqueeze(0)  # Add a batch dimension

    # # Convert the tensor to numpy array and reshape it for JSON serialization
    # x = train_data_point.cpu().detach().numpy().reshape([-1]).tolist()
    # data = {'input_data': [x]}

    return model, train_data_point.size(), train_data_point
