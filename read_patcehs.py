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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.preprocessing.image import ImageDataGenerator
####### PERCORSO IN LOCALE #########
path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))


tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
for i in range (0,len(tmp2)):
    for r in range(0,64):
        for c in range (0,64):
            if tmp2[i,r,c,:]!=1 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=5:
                print(tmp2[i,r,c,:])

# print(tmp2[500,:,:,:])
# print(tmp2[740,:,:,:])