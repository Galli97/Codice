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
# path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
path = r"C:\Users\Mattia\Desktop\Tentativi128\image_patches_TEST.npy"
path1 =  r"C:\Users\Mattia\Desktop\Tentativi128\label_patches_TEST.npy"
path2 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions.npy"

tmp1 = get_np_arrays(path)
tmp2 = get_np_arrays(path1)
predictions = get_np_arrays(path2)

SHAPE=128;

true = decode_masks(tmp2,SHAPE)
predictions = decode_predictions(predictions,SHAPE)
prediction = decode_masks(predictions,SHAPE)



foto = random.randint(0,len(tmp1)-1)
print('foto: ',foto)

soil_count=0;
bedrock_count=0;
sand_count=0;
bigrock_count=0;
null_count=0;

for r in range(0,SHAPE):
    for c in range (0,SHAPE):
        if tmp2[foto,r,c,:]==4:
            soil_count+=1
        elif tmp2[foto,r,c,:]==1:
            bedrock_count+=1
        elif tmp2[foto,r,c,:]==2:
            sand_count+=1
        elif tmp2[foto,r,c,:]==3:
            bigrock_count+=1
        elif tmp2[foto,r,c,:]==0:
            null_count+=1
        else:
            print(i)
            
print('null: ', null_count)
print('bedrock: ', bedrock_count)
print('sand: ', sand_count)
print('bigrock: ', bigrock_count)
print('soil: ', soil_count)


img=cv2.resize(tmp1[foto],(512,512))
img_prediction=cv2.resize(prediction[foto],(512,512))
label=cv2.resize(true[foto],(512,512))


cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.imshow('label',label )
cv2.waitKey(0) 
cv2.imshow('predictions',img_prediction)
cv2.waitKey(0) 