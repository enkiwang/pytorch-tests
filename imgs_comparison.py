# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 19:20:29 2018
show original vs reconstructed images
imgs_rec.data: 
@author: yongweiw
"""

import matplotlib.pyplot as plt
from img_utils import display_imgs

# ##imgs_rec (torch.Tensor)
idx = 0
idx_ = 100
# ##show a single image
#img = imgs_rec.data[idx,0,:,:].view(28,28)   #torch.Size([128, 1, 28, 28])
#img_ = imgs_org.data[idx,:].view(28,28)      #torch.Size([128,784])
##img = img.reshape([28,28])
#plt.figure('rec')
#plt.imshow(img,cmap='gray')
#plt.show()
#
#plt.figure('original')
#plt.imshow(img_,cmap='gray')
#plt.show()

# ##show multiple images
imgs = imgs_rec.data[idx:idx_,0,:,:].view(-1, 28*28)
imgs_ = imgs_org.data[idx:idx_,:]   #torch.Size([128,784])

numPerRow,numPerCol = 10, 10
imgH, imgW = 28, 28

#show image comparison
plt.figure('rec')
display_imgs(imgs,numPerRow,numPerCol,imgH=28,imgW=28)

plt.figure('original')
display_imgs(imgs_,numPerRow,numPerCol,imgH=28,imgW=28)