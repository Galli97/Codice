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
from glob import glob
from scipy.io import loadmat
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

##############Dataset##########
IMAGE_SIZE = 64
BATCH_SIZE = 4
NUM_CLASSES = 5
DATA_DIR = "./content/drive/MyDrive/Tesi/Dataset"
NUM_TRAIN_IMAGES = 100
NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(DATA_DIR, "Train_images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Train_labels/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Train_images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Train_labels/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)
##########################

tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(len(tmp1))
#print(tmp1.shape)
#print(tmp1)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(len(tmp2))
#print(tmp2.shape)
#print(tmp2)

# train_set = int(len(tmp1)*80/100)

# list_train = tmp1[:train_set]
# list_validation = tmp1[:train_set]

# label_train = tmp2[:train_set]
# label_validation = tmp2[:train_set]

shape=(64,64,3)

#model = rete(input_shape=shape,weight_decay=0.0001, classes=5)
model = DeeplabV3Plus(image_size=64,num_classes=5)

# x_train = data_generator(list_train,label_train)
# x_validation = data_generator(list_validation,label_validation)

optimizer = SGD(learning_rate=0.001, momentum=0.9)
loss_fn = keras.losses.SparseCategoricalCrossentropy()


model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])
#model.summary()
model.fit(x = train_dataset,batch_size = 4,epochs=25,steps_per_epoch=25,validation_data=val_dataset,validation_steps=25,validation_batch_size=4)