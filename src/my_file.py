# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2021 - 2023
------------------------------------------------------------------------------------------------------------------------
@Author: Diego Gigena Ivanovich - diego.gigena-ivanovich@silicon-austria.com
@File:   tutorial.py
@Time:   2/27/2023 - 9:55 PM
@IDE:    PyCharm
@desc:
------------------------------------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import Net
import os


if __name__ == '__main__': # -> Necesario solo para ejecutar en windows.
    ########################################################################################################################
    # Parámetros
    ########################################################################################################################
    print('Hola')
