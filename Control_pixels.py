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

path = r"C:\Users\Mattia\Documenti\Github\Codice\cropped_images.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\cropped_labels.npy"



### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print('tmp1: ',tmp1.shape)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
print('tmp2: ',tmp2.shape)

print(tmp1[8])
print(tmp2[8])

soil_count=0;
bedrock_count=0;
sand_count=0;
bigrock_count=0;
null_count=0;

SHAPE=128;
for i in range (0,len(tmp2)):
    for r in range(0,SHAPE):
        for c in range (0,SHAPE):
            # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
            #     print(tmp1[i,r,c,:])
            if tmp2[i,r,c,:]==0:
                soil_count+=1
            elif tmp2[i,r,c,:]==1:
                bedrock_count+=1
            elif tmp2[i,r,c,:]==2:
                sand_count+=1
            elif tmp2[i,r,c,:]==3:
                bigrock_count+=1
            elif tmp2[i,r,c,:]==255:
                null_count+=1
            else:
                print(tmp2[i,r,c,:])

print('null: ', null_count)
print('bedrock: ', bedrock_count)
print('sand: ', sand_count)
print('bigrock: ', bigrock_count)
print('soil: ', soil_count)

Pixels=null_count+bedrock_count+sand_count+bigrock_count+soil_count
pix=128*128*len(tmp2)

print('Pixel totali: ', pix)
print('Pixel labellati: ', Pixels)

