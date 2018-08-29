# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:25:29 2018
x.SignFun() function:
    forward: using sign function to give {-1,1}
    backward: re-defining sign grad based on input x,ie.,if x \in[-1,1], grad=1; 0 otherwise.

@author: yongweiw
"""

import torch
from torch.autograd import Function

class SignFun(Function):
    @staticmethod
    def forward(ctx,x):
        ctx.x = x
        return x.sign()
        
    @staticmethod
    def backward(ctx,grad_output):
        if ctx.x > 1 or ctx.x < -1:
            return 0 * grad_output
        else:
            return 1 * grad_output


sign_test = SignFun.apply

#### data x
x = torch.randn(1,requires_grad=True)

#### out = sign(x)
out = sign_test(x) 

print(x)
#print(out)

out.backward()

print(x.grad)
# x=-0.6212; grad as defined: x.grad= 1 since within bound (-1,1)
# x=1.9424; grad as defined: x.grad= 0 since out of bound (-1,1)



