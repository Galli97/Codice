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
path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
path2 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions.npy"

tmp1 = get_np_arrays(path)
tmp2 = get_np_arrays(path1)
predictions = get_np_arrays(path2)

true = decode_masks(tmp2)
predictions = decode_predictions(predictions)
prediction = decode_mask(predictions)



foto = random.randint(0,len(predictions)-1)

img=cv2.resize(tmp1[foto],(1024,1024))
img_true=cv2.resize(true[foto],(1024,1024))
img_prediction=cv2.resize(prediction[foto],(1024,1024))

cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.imshow('label',img_true )
cv2.waitKey(0) 
cv2.imshow('predictions',img_prediction)
cv2.waitKey(0) 