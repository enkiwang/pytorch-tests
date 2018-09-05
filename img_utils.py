# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:13:41 2018
display_imgs: display images row * col
@author: yongweiw
"""

import numpy as np
import matplotlib.pyplot as plt

def display_imgs(imgs,numPerRow,numPerCol,imgH=32,imgW=32):
    Img = np.zeros([imgH*numPerCol, imgW*numPerRow])
    
    for i in range(numPerCol):
        for j in range(numPerRow):
            tmp = imgs[i*numPerRow+j,:].reshape([imgH,imgW])
            Img[i*imgW:(i+1)*imgW, j*imgH:(j+1)*imgH] = tmp
            
    plt.imshow(Img,cmap='gray')
    plt.show()
    
