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
        self.pool_w_stride = nn.MaxPool2d(2, 2)
        self.pool_wo_stride = nn.MaxPool2d(2, 1)

        conv1_channels = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_channels, kernel_size=3)
        self.batch1 = nn.BatchNorm2d(conv1_channels)

        conv2_channels = 64
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=2)
        self.batch2 = nn.BatchNorm2d(conv2_channels)

        conv3_channels = 128
        self.conv3 = nn.Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(conv3_channels)

        self.fc1 = nn.Linear(6 * 6 * 128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        x = self.pool_wo_stride(self.relu(self.batch1(self.conv1(img))))
        # self.conv1(img) 32-3+1 = 30
        # self.pool(2,2) 30-1 = 29
        x = self.pool_w_stride(self.relu(self.batch2(self.conv2(x))))
        # self.conv1(img) 29-2+1 = 28
        # self.pool(img) 28/1 = 14
        x = self.pool_w_stride(self.relu(self.batch3(self.conv3(x))))
        # self.conv1(img) 14-2+1 = 12
        # self.pool(img) 12/2 = 6
        # x = torch.flatten(x, 1)
        x = x.view(-1, 6 * 6 * 128)
        # print(x.size())
        # 16 channels 6*6
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
