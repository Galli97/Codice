import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import image
import cv2
import numpy as np
import os
from PIL import Image
from rete import *
from tensorflow.keras.optimizers import SGD
from utils import *
from keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.utils import shuffle
from sklearn.feature_extraction import image

path = r"C:\Users\Mattia\Desktop\Tentativi128\image_patches_TEST.npy"
path1 =  r"C:\Users\Mattia\Desktop\Tentativi128\label_patches_TEST.npy"
path2 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions.npy"

tmp1 = get_np_arrays(path)
tmp2 = get_np_arrays(path1)
predictions = get_np_arrays(path2)

tmp1=tmp1[:64]
tmp2=tmp2[:64]
predictions=predictions[:64]

SHAPE=128;

tmp2 = decode_masks(tmp2,SHAPE)

predictions = decode_predictions(predictions,SHAPE)
predictions = decode_masks(predictions,SHAPE)

overlay=[];

for i in range (len(tmp2)):
    true_image = tmp1[i]
    true_image = np.asarray(true_image, np.float64)
    label_img = tmp2[i]
    label_img = np.asarray(label_img, np.float64)
    overlay_img = cv2.addWeighted(true_image, 1, label_img, 0.001, 0)
    overlay.append(overlay_img)

# true_image=tmp1[5]
# true_image = np.asarray(true_image, np.float64)
# print(true_image.shape)
# immagine=cv2.resize(true_image,(512,512))
# print(immagine.shape)
# cv2.imshow('image', immagine) 
# cv2.waitKey(0) 

# label_img=cv2.resize(true[5],(512,512))
# label_img = np.asarray(label_img, np.float64)
# print(label_img.shape)
# cv2.imshow('image', label_img) 
# cv2.waitKey(0) 

# overlay = cv2.addWeighted(immagine, 1, label_img, 0.0008, 0)
# cv2.imshow('overlay',overlay)
# cv2.waitKey(0) 


img1 = np.hstack((overlay[0], overlay[1],overlay[2], overlay[3],overlay[4], overlay[5],overlay[6], overlay[7]))
img2 = np.hstack((overlay[0+8],overlay[1+8], overlay[2+8],overlay[3+8], overlay[4+8],overlay[5+8], overlay[6+8],overlay[7+8]))
img3 = np.hstack((overlay[0+2*8],overlay[1+2*8], overlay[2+2*8],overlay[3+2*8], overlay[4+2*8],overlay[5+2*8], overlay[6+2*8],overlay[7+2*8]))
img4 = np.hstack((overlay[0+3*8],overlay[1+3*8], overlay[2+3*8],overlay[3+3*8], overlay[4+3*8],overlay[5+3*8], overlay[6+3*8],overlay[7+3*8]))
img5 = np.hstack((overlay[0+4*8],overlay[1+4*8], overlay[2+4*8],overlay[3+4*8], overlay[4+4*8],overlay[5+4*8], overlay[6+4*8],overlay[7+4*8]))
img6 = np.hstack((overlay[0+5*8],overlay[1+5*8], overlay[2+5*8],overlay[3+5*8], overlay[4+5*8],overlay[5+5*8], overlay[6+5*8],overlay[7+5*8]))
img7 = np.hstack((overlay[0+6*8],overlay[1+6*8], overlay[2+6*8],overlay[3+6*8], overlay[4+6*8],overlay[5+6*8], overlay[6+6*8],overlay[7+6*8]))
img8 = np.hstack((overlay[0+7*8],overlay[1+7*8], overlay[2+7*8],overlay[3+7*8], overlay[4+7*8],overlay[5+7*8], overlay[6+7*8],overlay[7+7*8]))

result = np.vstack((img1,img2,img3,img4,img5,img6,img7,img8))

cv2.imshow('overlay',result)
cv2.waitKey(0)
