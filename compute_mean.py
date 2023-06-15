import cv2,torch
import numpy as np
import  pandas as pd
import os

# 读数据

train_text_path = os.path.join('','','Dataset/train_text')

num_picture = 2000
img_w,img_h = 32,32
imgs = np.empty([img_w,img_h,3,1])
mean = [] #均值
std = [] #方差

with open(train_text_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip().split()[0]
        img = cv2.imread(line[0])
        img = cv2.resize(img,[img_w,img_h])

        img = img[:,:,:,np.newaxis]
        imgs = np.concatenate((imgs,img),axis = 3)

for i in range(3):
    pixel = img[:,:,i,:].reval()
    means.append(np.mean(pixel))
    stds.append(np.std(pixel))

    #在这里获得了所有图像在每个通道的均值和方差，接下来进行归一化

means.reverse()
stds.reverse()

#这里获得数据可以进行btatch_normalization

