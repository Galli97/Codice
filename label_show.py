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
path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"

tmp1 = get_np_arrays(path)
tmp2 = get_np_arrays(path1)

print(tmp2[10])
true = decode_masks(tmp2)



foto = random.randint(0,len(tmp1)-1)
print('foto: ',foto)

img=cv2.resize(true[foto],(512,512))
img_prediction=cv2.resize(true[10],(512,512))
label=cv2.resize(true[5],(512,512))


cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.imshow('label',label )
cv2.waitKey(0) 
cv2.imshow('predictions',img_prediction)
cv2.waitKey(0) 