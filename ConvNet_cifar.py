#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:15:06 2018
ConvNet (LeNet) for classification tasks, tested on CIFAR-10 dataset

For GPU computations:
    1)put the MODEL (net .obj) onto the specified computing device
    2)put the DATA onto the specified computing device

specifications:
    1)GPU: NVIDIA Titan X (4 GPUs)
    2)pytorch version: 0.40 (torch.device not available for lower pytorch versions)
loss: 2.306 (epc=1), 0.94 (epc=50), 0.579(epc=100),..., 0.178(epc=200),.., 0.000(epc=520)
    
Note that:
 This code will display "Exception NameError", due to 0.40 pytorch version. Downgrade to 
 0.30 with python2.7 will settle this error message. 
 Anyway, this error message will only display at the end of training.     
    
@author: @ https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
modified by: @yongwei
"""


import torch
import torchvision
import torchvision.transforms as transforms

"""
specify computing device: cuda:0 or cpu
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Your computing device is",device)

## **************** prepare CIFAR10 dataset ****************

batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# cifar10 dataset has been downloaded, and put in /bigdata/yongwei/data/cifar10/
trainset = torchvision.datasets.CIFAR10(root='/bigdata/yongwei/data/cifar10/', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size ,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/bigdata/yongwei/data/cifar10/', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size ,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


## **************** show cifar-10 data samples ****************

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print("")
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print("")

## **************** Define a ConvNet ****************

import torch.nn as nn
import torch.nn.functional as F##### for multiple GPUs, use this code:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # input chan=3,output chan=6, kernel_size = 5*5
        self.pool = nn.MaxPool2d(2, 2)   # 32*32 -> 16*16
        self.conv2 = nn.Conv2d(6, 16, 5) # input chan=6, output chan=16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # affine projection, input dim=16*5*5, output dim=120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # first layer: Conv(x,W) -> Relu ->max-pooling 
        x = self.pool(F.relu(self.conv2(x)))   # second layer: Conv(x,W) -> Relu ->max-pooling 
        x = x.view(-1, 16 * 5 * 5)             # Fc layer: affine projection. 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

##### for multiple GPUs, use this code:
if torch.cuda.device_count() > 1:
    print("We are using", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)  # specify multiple GPU computing

# put net .obj (MODEL) onto the specified computing device on single GPU
net.to(device)  # loading net (MODEL) to your device(s), either cpu or GPU(s)

## **************** Define a loss function + optimizer ****************

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## **************** Train the ConvNet ****************
epoch_ = 800
disp_ = 50  # display loss per disp_ batches 
for epoch in range(epoch_):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # put data onto the specified computing device
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % disp_ == 0:    # print losses every disp_ mini-batches (e.g.,disp_=20: per 20*128=2560 training samples)
            print('epoch %d / %d, minibatch_num  %d,  loss: %.3f' %
                  (epoch + 1, epoch_, i + 1, running_loss / disp_))
            running_loss = 0.0        
            
    if (epoch+1) % 20 == 0:
        torch.save(net.state_dict(), "LeNet_model/LeNet_cifar_{0:04d}.pt".format(epoch+1))
        print(" LeNet model saved at %3d th epoch!" % (epoch+1))
       
print('\n Finished Training! \n')
