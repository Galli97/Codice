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


x_train = datagenerator(tmp1,tmp2,2)

optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn = keras.losses.SparseCategoricalCrossentropy()


model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])
model.summary()
model.fit(x = x_train,batch_size = 32,epochs=25,steps_per_epoch=6)