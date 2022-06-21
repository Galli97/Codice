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
path = r"/content/drive/MyDrive/Tesi/image_arrays.npy"
path1 = r"/content/drive/MyDrive/Tesi/label_arrays.npy"

tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 


tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file

#### DEFINISCO ALCUNE VARIABILI UTILI #####
shape=(64,64,3)
EPOCHS=8
BATCH=16
train_set = int((tmp1.shape[0])*(2/3))
steps = 3 #int(train_set/EPOCHS)

### DEFINISCO IL MODELLO ########
model = rete(input_shape=shape,weight_decay=0., classes=5)
#model = DeeplabV3Plus(image_size=64, num_classes=4)

##### SPLITTO IL TRAIN SET PER CREARE UN VALIDATION SET ####

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]

#### PASSO LE LISTE IN DATAGENERATOR CHE MI PREPARA I DATI PER PASSARE NELLA RETE #######
x_train = datagenerator(list_train,label_train,BATCH)
x_validation = datagenerator(list_validation,label_validation,BATCH)

### DEFINISCO OPTIMIZER E LOSS  ######
optimizer = SGD(learning_rate=0.001, momentum=0.)#,clipnorm=1.0)
loss_fn = keras.losses.CategoricalCrossentropy()

### ESEGUO IL COMPILE ###
model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])

## IL SUMMARY DA N OUTPUT LA STRUTTURA DEL MODELLO CON I VARI LAYERS ####
model.summary()

## AVVIO IL TRAINING ###
model.fit(x = x_train,batch_size = BATCH,epochs=EPOCHS,steps_per_epoch=steps,validation_data=(list_validation, label_validation),validation_steps=steps,validation_batch_size=BATCH)