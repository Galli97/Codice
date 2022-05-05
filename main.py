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
#print(len(tmp1))
#print(tmp1.shape)
#print(tmp1)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(len(tmp2))
#print(tmp2.shape)
#print(tmp2)

shape=(64,64,3)

model = rete(input_shape=shape,weight_decay=0., classes=5)

EPOCHS=100
train_set = int(len(image_list)*(2/3))
steps = int(train_set/EPOCHS)

list1_train = tmp1[:train_set]
list2_train = tmp2[:train_set]


list1_test = tmp1[train_set:]
list2_test = tmp2[train_set:]

x_train = datagenerator(list1_train,list2_train,32)
x_test = datagenerator(list1_test,list2_test,32)

optimizer = SGD(learning_rate=0.001, momentum=0.9)
loss_fn = keras.losses.CategoricalCrossentropy()



model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])
model.summary()
model.fit(x = x_train,epochs=EPOCHS,steps_per_epoch=steps,validation_data = x_test,validation_steps=steps,validation_batch_size=32)