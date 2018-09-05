# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:25:15 2018
showing stacks of images as row*col
@author: yongweiw
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def read_img_file(root,img_size=256*256):
    imgs_list = os.listdir(root)
    imgs_list.sort()
    imgs = np.zeros([len(imgs_list),img_size])
    for i in range(len(imgs_list)):
        img = plt.imread(root + imgs_list[i])
        imgs[i,:] = img.ravel()
    return imgs

def display_imgs(imgs,numPerRow,numPerCol,imgH=32,imgW=32):
    Img = np.zeros([imgH*numPerCol, imgW*numPerRow])
    
    for i in range(numPerCol):
        for j in range(numPerRow):
            tmp = imgs[i*numPerRow+j,:].reshape([imgH,imgW]) #previously I made a mistake here
            Img[i*imgW:(i+1)*imgW, j*imgH:(j+1)*imgH] = tmp

    plt.imshow(Img,cmap='gray')
    plt.axis('off')
    plt.show()

root = '/path/to/file1/'
root_gt = '/path/to/file2'

imgs = read_img_file(root,img_size=256*256)
imgs_gt = read_img_file(root_gt,img_size=256*256)


display_imgs(np.vstack((imgs,imgs_gt)),numPerRow=8,numPerCol=2,imgH=256,imgW=256)

