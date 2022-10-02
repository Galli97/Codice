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
# ####### PERCORSO IN LOCALE #########
path = r"C:\Users\Mattia\Documenti\Github\Codice\crop_Perseverance.npy"
path1 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions_crop.npy"

tmp1 = get_np_arrays(path)
predictions = get_np_arrays(path1)

print(len(predictions))
print(predictions[0].shape)

tmp1=tmp1[576:640]
predictions=predictions[576:640]

SHAPE=128;

predictions = decode_predictions(predictions,SHAPE)
prediction = decode_masks(predictions,SHAPE)

foto = random.randint(0,50)
print('foto: ',foto)

# img=cv2.resize(tmp1[foto],(512,512))
# img_prediction=cv2.resize(prediction[foto],(512,512))

# cv2.imshow('image', img) 
# cv2.waitKey(0) 

# cv2.imshow('predictions',img_prediction)
# cv2.waitKey(0) 

# img=cv2.resize(tmp1[int(foto/2)],(512,512))
# img_prediction=cv2.resize(prediction[int(foto/2)],(512,512))

# cv2.imshow('image', img) 
# cv2.waitKey(0) 
 
# cv2.imshow('predictions',img_prediction)
# cv2.waitKey(0) 

# img=cv2.resize(tmp1[7],(512,512))
# img_prediction=cv2.resize(prediction[7],(512,512))


overlay=[];

for j in range (64):
    true_image = tmp1[j]
    true_image = np.asarray(true_image, np.float32)
    label_img = prediction[j]
    label_img = np.asarray(label_img, np.float32)
    overlay_img = cv2.addWeighted(true_image, 0.9, label_img,  0.0018, 0)
    overlay.append(overlay_img)


img1 = np.hstack((overlay[0], overlay[1],overlay[2], overlay[3],overlay[4], overlay[5],overlay[6], overlay[7]))
img2 = np.hstack((overlay[0+8],overlay[1+8], overlay[2+8],overlay[3+8], overlay[4+8],overlay[5+8], overlay[6+8],overlay[7+8]))
img3 = np.hstack((overlay[0+2*8],overlay[1+2*8], overlay[2+2*8],overlay[3+2*8], overlay[4+2*8],overlay[5+2*8], overlay[6+2*8],overlay[7+2*8]))
img4 = np.hstack((overlay[0+3*8],overlay[1+3*8], overlay[2+3*8],overlay[3+3*8], overlay[4+3*8],overlay[5+3*8], overlay[6+3*8],overlay[7+3*8]))
img5 = np.hstack((overlay[0+4*8],overlay[1+4*8], overlay[2+4*8],overlay[3+4*8], overlay[4+4*8],overlay[5+4*8], overlay[6+4*8],overlay[7+4*8]))
img6 = np.hstack((overlay[0+5*8],overlay[1+5*8], overlay[2+5*8],overlay[3+5*8], overlay[4+5*8],overlay[5+5*8], overlay[6+5*8],overlay[7+5*8]))
img7 = np.hstack((overlay[0+6*8],overlay[1+6*8], overlay[2+6*8],overlay[3+6*8], overlay[4+6*8],overlay[5+6*8], overlay[6+6*8],overlay[7+6*8]))
img8 = np.hstack((overlay[0+7*8],overlay[1+7*8], overlay[2+7*8],overlay[3+7*8], overlay[4+7*8],overlay[5+7*8], overlay[6+7*8],overlay[7+7*8]))

result = np.vstack((img1,img2,img3,img4,img5,img6,img7,img8))

result = cv2.resize(result, (512, 512))
cv2.imshow('overlay',result)
cv2.waitKey(0)