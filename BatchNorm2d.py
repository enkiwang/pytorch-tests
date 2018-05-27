#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:26:52 2018
torch.nn.BatchNorm2d():
    torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, 
                     track_running_stats=True)
        where num_features means the input data channel C (see below).
        
"The mean and standard-deviation are calculated per-dimension over the mini-batches."

    Input data: (N, C, H, W) and output has the same dim. as the input.

Details refer to, https://pytorch.org/docs/stable/nn.html#batchnorm2d
@author: yongweiw
"""

import torch
import torch.nn as nn

# input DATA parameters
N = 1
C_in = 1
H_in = 3
W_in = 3

# data generation
torch.manual_seed(1000)
data = torch.randn(N, C_in, H_in, W_in)

# bn with learnable parameters
bn = nn.BatchNorm2d(C_in)
data_bn = bn(data)

# bn without learnable parameters
bn2 = nn.BatchNorm2d(C_in, affine=False)
data_bn2 = bn2(data)

# print data
print(data)
print(data_bn)
print(data_bn2)

"""
tensor([[[[-1.1720, -0.3929,  0.5265],
          [ 1.1065,  0.9273, -1.7421],
          [-0.7699,  0.7864, -1.9963]]]])
tensor(1.00000e-02 *
       [[[[-1.0147, -0.1050,  0.9684],
          [ 1.6456,  1.4363, -1.6802],
          [-0.5452,  1.2719, -1.9770]]]])
tensor([[[[-0.7758, -0.0803,  0.7404],
          [ 1.2582,  1.0982, -1.2846],
          [-0.4168,  0.9724, -1.5115]]]])

"""