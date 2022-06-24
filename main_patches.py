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
###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
path = r"/content/drive/MyDrive/Tesi/image_patches.npy"
path1 = r"/content/drive/MyDrive/Tesi/label_patches.npy"

# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))
print(tmp1.shape)




tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)
print(len(tmp2))

# N = len(tmp2)
# tmp1=tmp1[:N]
# print(tmp1.shape)

#### PRENDO UNA PARTE DEL DATASET (20%) E LO UTILIZZO PER IL VALIDATION SET #####
train_set = int(len(tmp2)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]
print('list_train: ',list_train.shape)
print('list_validation: ',list_validation.shape)

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]
print('label_train: ',label_train.shape)
print('label_validation: ',label_validation.shape)

# soil:  674741
# bedrock:  1031730
# sand:  378918
# bigrock:  174845
# null:  1835766
soil_pixels = 674741;
bedrock_pixels = 1031730;
sand_pixels = 378918;
bigrock_pixels = 174845;
null_pixels = 1835766;
PIXELS=soil_pixels+bedrock_pixels + sand_pixels+bigrock_pixels#+null_pixels ;
loss_weights=[soil_pixels/PIXELS,bedrock_pixels/PIXELS,sand_pixels/PIXELS,bigrock_pixels/PIXELS,0]
# label_train = label_train.reshape((len(label_train),64*64,1))
# label_validation = label_validation.reshape((len(label_validation),64*64,1))
# print('label_train: ',label_train.shape)
# print('label_validation: ',label_validation.shape)

###### DEFINISCO IL MODELLO #######
shape=(64,64,1)
print(shape)
BATCH= 32
EPOCHS=20
steps = 5#int(train_set/EPOCHS)
weight_decay = 0.0001/2
batch_shape=(BATCH,64,64,1)
model = rete(input_shape=shape,weight_decay=weight_decay,batch_shape=None, classes=5)

#model = DeeplabV3Plus(image_size=64,num_classes=5)

##### USO DATAGENERATOR PER PREPARARE I DATI DA MANDARE NELLA RETE #######
x_train = datagenerator(list_train,label_train,BATCH)
x_validation = datagenerator(list_validation,label_validation,BATCH)
#print(type(x_train))

# sample_weight = np.ones(shape=(len(label_train),64,64))
# print(sample_weight.shape)
# sample_weight[:,0] = 1.5
# sample_weight[:,1] = 0.5
# sample_weight[:,2] = 1.5
# sample_weight[:,3] = 3.0
# sample_weight[:,4] = 0

# val_sample_weight = np.ones(shape=(len(label_validation),64,64))
# print(val_sample_weight.shape)
# val_sample_weight[:,0] = 1.5
# val_sample_weight[:,1] = 0.5
# val_sample_weight[:,2] = 1.5
# val_sample_weight[:,3] = 3.0
# val_sample_weight[:,4] = 0

# Create a Dataset that includes sample weights
# (3rd element in the return tuple).
# x_train = tf.data.Dataset.from_tensors((list_train, label_train, sample_weight))
# x_validation = tf.data.Dataset.from_tensors((list_validation, label_validation, val_sample_weight))

# Shuffle and slice the dataset.
# x_train = x_train.batch(BATCH)
# x_validation=x_validation.batch(BATCH)
#### DEFINSICO I PARAMETRI PER IL COMPILE (OPTIMIZER E LOSS)

lr_base = 0.01 * (float(BATCH) / 16)
optimizer = SGD(learning_rate=lr_base, momentum=0.)
#optimizer=keras.optimizers.Adam(learning_rate=0.001)
loss_fn =keras.losses.SparseCategoricalCrossentropy()#keras.losses.SparseCategoricalCrossentropy(from_logits=True) #iou_coef #softmax_sparse_crossentropy_ignoring_last_label

model.compile(optimizer = optimizer, loss = loss_fn , metrics =[tf.keras.metrics.SparseCategoricalAccuracy()],loss_weights=loss_weights)#,sample_weight_mode='temporal'))#[tf.keras.metrics.MeanIoU(num_classes=5)])#['accuracy'])#[sparse_accuracy_ignoring_last_label])#,sample_weight_mode='temporal')

### AVVIO IL TRAINING #####
model.summary()
model.fit(x = x_train,batch_size = BATCH,epochs=EPOCHS,steps_per_epoch=steps,validation_data=x_validation,validation_steps=steps,validation_batch_size=BATCH)