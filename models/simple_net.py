# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2021 - 2023
------------------------------------------------------------------------------------------------------------------------
@Author: Diego Gigena Ivanovich - diego.gigena-ivanovich@silicon-austria.com
@File:   nn_models.py
@Time:   2/27/2023 - 10:26 PM
@IDE:    PyCharm
@desc:
------------------------------------------------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, batch_size):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)
        self.fc1 = nn.Linear(27 * 27 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        x = self.relu(self.conv1(img))
        # self.conv1(img) 32-3+1 = 30
        x = self.relu(self.conv2(x))
        # self.conv1(img) 30-4+1 = 27
        # print(x.size())
        # x = torch.flatten(x, 1)
        x = x.view(-1, 27*27*16)
        # print(x.size())
        # 16 channels 6*6
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
