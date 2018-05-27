#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 07:23:40 2018
1.torch.nn.Conv2d(): 
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                    padding=0,dilation=1,groups=1,bias=True)
    DATA INPUT:(N,C_in,H_in,W_in) # N-C-H-W
    DATA OUTPUT:(N,C_out,H_out,W_out) # N-C-H-W
    where, H_out = ground_round[(H_in + 2*padding[0] -dialation[0]*(kernel_size[0]-1)-1)/stride[0] + 1]
    W_out follows the same rule.
    
2.torch.nn.ConvTranspose2d():
    torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
                            padding=0,dilation=1,groups=1,bias=True)
    DATA INPUT: (N, C_in, H_in, W_in), and OUTPUT: (N,C_out,H_out,W_out) # N-C-H-W
    where, H_out = (H_in - 1)*stride[0] - 2*padding[0] + kernel_size[0] + output_padding[0]
    
Summary: data format N-C-H-W; conv.() format i-o-k-s-p.
@author: yongweiw
"""

import torch
import torch.nn as nn

#N = 50  # number of data samples
#C_in = 3
#H_in = 32
#W_in = 32

# input DATA parameters
N = 1
C_in = 1
H_in = 3
W_in = 3

# convolution parameters
in_channels = C_in
out_channels = 8
k = 3
s = 1
p = 0

# data generation
torch.manual_seed(1000)
data1 = torch.randn(N, C_in, H_in, W_in)

""" **************************** torch.nn.Conv2d(in_chan,out_chan,k,s,p) **************************** """
conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)  #conv. para. (1, 8, 3, 1, 0)
data1_conv = conv(data1)    #data1_conv 

print("data1.size():", data1.size())      #(N, 1, 3, 3)
print("data1_conv.size():", data1_conv.size()) #(N, 8, 1, 1)

""" **************************** torch.nn.ConvTranspose2d(in_chan,out_chan,k,s,p) **************************** """
transposed_conv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=k, stride=s, padding=p)

# output_size must be specified to determine real paddings, otherwise feature maps will be getting smaller
# after transposed_convolution (convolution in essence)
data1_conv_tconv = transposed_conv (data1_conv, output_size=data1.size())

print("data1_conv_tconv.size():", data1_conv_tconv.size())

print(data1)
print(data1_conv_tconv)

"""
('data1.size():', (1, 1, 3, 3))
('data1_conv.size():', (1, 8, 1, 1))
('data1_conv_tconv.size():', (1, 1, 3, 3))
tensor([[[[-1.1720, -0.3929,  0.5265],
          [ 1.1065,  0.9273, -1.7421],
          [-0.7699,  0.7864, -1.9963]]]])
tensor([[[[ 0.0376, -0.2292, -0.0740],
          [-0.3514, -0.3656,  0.2330],
          [ 0.2754,  0.0635, -0.1715]]]])

data1 does not equal to data1_conv_tconv;
Unlike that in TF, conv. or tconv. weights have been initialized in pytorch?
"""
