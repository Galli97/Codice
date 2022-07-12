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
path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"
# path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))
print(tmp1.shape)

soil_count=0;
bedrock_count=0;
sand_count=0;
bigrock_count=0;
null_count=0;

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)

train_set = int(len(tmp2)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]
print('list_train: ',list_train.shape)
print('list_validation: ',list_validation.shape)

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]
print('label_train: ',label_train.shape)
print('label_validation: ',label_validation.shape)

# for i in range (0,len(tmp2)):
#     for r in range(0,64):
#         for c in range (0,64):
#             # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
#             #     print(tmp1[i,r,c,:])
#             if tmp2[i,r,c,:]==4:
#                 soil_count+=1
#             elif tmp2[i,r,c,:]==1:
#                 bedrock_count+=1
#             elif tmp2[i,r,c,:]==2:
#                 sand_count+=1
#             elif tmp2[i,r,c,:]==3:
#                 bigrock_count+=1
#             elif tmp2[i,r,c,:]==0:
#                 null_count+=1
#             else:
#                 print(i)

for i in range (0,len(label_train)):
    for r in range(0,64):
        for c in range (0,64):
            # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
            #     print(tmp1[i,r,c,:])
            if label_train[i,r,c,:]==4:
                soil_count+=1
            if label_train[i,r,c,:]==1:
                bedrock_count+=1
            if label_train[i,r,c,:]==2:
                sand_count+=1
            if label_train[i,r,c,:]==3:
                bigrock_count+=1
            if label_train[i,r,c,:]==0:
                null_count+=1

print('null: ', null_count)
print('bedrock: ', bedrock_count)
print('sand: ', sand_count)
print('bigrock: ', bigrock_count)
print('soil: ', soil_count)



soil_count_val=0;
bedrock_count_val=0;
sand_count_val=0;
bigrock_count_val=0;
null_count_val=0;

for i in range (0,len(label_validation)):
    for r in range(0,64):
        for c in range (0,64):
            # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
            #     print(tmp1[i,r,c,:])
            if label_validation[i,r,c,:]==4:
                soil_count_val+=1
            if label_validation[i,r,c,:]==1:
                bedrock_count_val+=1
            if label_validation[i,r,c,:]==2:
                sand_count_val+=1
            if label_validation[i,r,c,:]==3:
                bigrock_count_val+=1
            if label_validation[i,r,c,:]==0:
                null_count_val+=1

print('null_val: ', null_count_val)
print('bedrock_val: ', bedrock_count_val)
print('sand_val: ', sand_count_val)
print('bigrock_val: ', bigrock_count_val)
print('soil_val: ', soil_count_val)