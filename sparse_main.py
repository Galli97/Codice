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

train_set = int(len(tmp1)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[:train_set]

label_train = tmp2[:train_set]
label_validation = tmp2[:train_set]

shape=(64,64,3)
BATCH=16
# sample_weight = np.zeros((153,4096))
# sample_weight[:, 0] += 0.41
# sample_weight[:, 1] += 1.87
# sample_weight[:, 2] += 1.1
# sample_weight[:, 3] += 7.05
# sample_weight[:, 4] += 7.05

#model = rete(input_shape=shape,weight_decay=0., classes=5)
model = DeeplabV3Plus(image_size=64,num_classes=5)

x_train = datagenerator(list_train,label_train,BATCH)
x_validation = datagenerator(list_validation,label_validation,BATCH)

optimizer = SGD(learning_rate=0.001, momentum=0.)
loss_fn = keras.losses.SparseCategoricalCrossentropy()


model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])#,sample_weight_mode='temporal')
#model.summary()
model.fit(x = x_train,batch_size = BATCH,epochs=25,steps_per_epoch=10,validation_data=(list_validation, label_validation),validation_steps=10,validation_batch_size=BATCH)