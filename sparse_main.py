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
####### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
path = r"/content/drive/MyDrive/Tesi/image_arrays_sparse.npy"
path1 = r"/content/drive/MyDrive/Tesi/label_arrays_sparse.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))


tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))

#### PRENDO UNA PARTE DEL DATASET (20%) E LO UTILIZZO PER IL VALIDATION SET #####
train_set = int(len(tmp1)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]

###### DEFINISCO IL MODELLO #######
shape=(64,64,3)
BATCH= 16
EPOCHS=5
steps = 5#int(train_set/EPOCHS)
model = rete(input_shape=shape,weight_decay=0., classes=5)
#model = DeeplabV3Plus(image_size=64,num_classes=5)

##### USO DATAGENERATOR PER PREPARARE I DATI DA MANDARE NELLA RETE #######
x_train = datagenerator(list_train,label_train,BATCH)
x_validation = datagenerator(list_validation,label_validation,BATCH)
#print(type(x_train))

#### DEFINSICO I PARAMETRI PER IL COMPILE (OPTIMIZER E LOSS)
optimizer = SGD(learning_rate=0.0001, momentum=0.)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])#,sample_weight_mode='temporal')

### AVVIO IL TRAINING #####
#model.summary()
model.fit(x = x_train,batch_size = BATCH,epochs=EPOCHS,steps_per_epoch=steps,validation_data=(list_validation, label_validation),validation_steps=steps,validation_batch_size=BATCH)