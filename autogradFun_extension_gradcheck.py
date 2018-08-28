# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:11:55 2018
torch.autograd:
1.extending torch.autograd.Function:
    1)forward(ctx,vags), the code that performs the operations
    2)backward(ctx,grad_output), gradient formula for differentiating the operation
Ref.https://pytorch.org/docs/master/notes/extending.html#adding-a-module
2. gradcheck, numerical gradient checking
@author: Ref.
"""

import torch
from torch.autograd import Function
from math import exp
#import torch.autograd.Function as Function

class Exp_test(Function):
    @staticmethod
    def forward(ctx,x_input):
        result = x_input.exp()
        ctx.save_for_backward(result)
        return result
        
    @staticmethod
    def backward(ctx,grad_output):
        result, = ctx.saved_tensors
        print(result)
        print(grad_output)
        return grad_output * result ##since d(exp(x))/dx = exp(x)
        
exp_test = Exp_test.apply
value = 3
x_input = torch.Tensor([value])
x_exp = exp_test(x_input)

## check that d(exp(x))/dx = exp(x)
print(x_exp)  #tensor([20.0855])
print(exp(value)) #20.085536923187668

class MulConst_test(Function):
    @staticmethod
    def forward(ctx,tensor,constant):
        ctx.constant = constant #non-tensor argument
        return tensor * constant
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None ##since d(ax)/dx = a
        


mul_test = MulConst_test.apply
## d(ax)/dx = a
var_1 = torch.Tensor([1])
var_2 = 5 #constant

x_mul = mul_test(var_1,var_2)
print(x_mul)


##### do gradient checking
from torch.autograd import gradcheck
var = torch.rand(1,requires_grad=True)
input_data = var
test = gradcheck(exp_test,input_data, eps=1e-6,atol=1e-4)
#numerical:tensor([1.1325])
#analytical:tensor([[1.1138]])

