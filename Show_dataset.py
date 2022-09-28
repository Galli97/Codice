import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import image
from keras.callbacks import *
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from rete import *
from tensorflow.keras.optimizers import SGD
from utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.preprocessing.image import ImageDataGenerator

# path = r"C:\Users\Mattia\Desktop\Dataset_1\255Data\Dataset_1_255\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Dataset_1\255Data\Dataset_1_255\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\Datase_BigRock\Data510\image_patches.npy"
# path1 = r"C:\Users\Mattia\Desktop\Datase_BigRock\Data510\label_patches.npy"

# path = r"C:\Users\Mattia\Desktop\Dataset_Resize\Selezionate2000\Data255\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Dataset_Resize\Selezionate2000\Data255\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\Resize512_crop128\Dataset_Res512\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Resize512_crop128\Dataset_Res512\final_labels.npy"

path = r"C:\Users\Mattia\Desktop\Resize256_crop128\Dataset_Res256\final_images.npy"
path1 = r"C:\Users\Mattia\Desktop\Resize256_crop128\Dataset_Res256\final_labels.npy"
### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print('tmp1: ',tmp1.shape)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
print('tmp2: ',tmp2.shape)

SHAPE=128
masks=decode_masks(tmp2[10:15],SHAPE)
tmp1=tmp1[10:15]
overlay=[];
for d in range (len(masks)):
    true_image = tmp1[d]
    true_image = np.asarray(true_image, np.float32)
    label_img = masks[d]
    label_img = np.asarray(label_img, np.float32)
    overlay_img = cv2.addWeighted(true_image, 1, label_img, 0.002, 0)
    overlay.append(overlay_img)

true_image = tmp1[3]
true_image = np.asarray(true_image, np.float32)
img=cv2.resize(true_image,(512,512))

label1=cv2.resize(masks[0],(512,512))
label2=cv2.resize(masks[1],(512,512))
label3=cv2.resize(masks[2],(512,512))
label4=cv2.resize(masks[3],(512,512))

image1=cv2.resize(overlay[0],(512,512))
image2=cv2.resize(overlay[1],(512,512))
image3=cv2.resize(overlay[2],(512,512))
image4=cv2.resize(overlay[3],(512,512))

cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('label', label1)
cv2.waitKey(0) 
cv2.imshow('image', image2)
cv2.waitKey(0) 
cv2.imshow('label', label2)
cv2.waitKey(0) 
cv2.imshow('image', image3)
cv2.waitKey(0) 
cv2.imshow('label', label3)
cv2.waitKey(0)  
cv2.imshow('image', image4)
cv2.waitKey(0) 
cv2.imshow('label', label4)
cv2.waitKey(0)  