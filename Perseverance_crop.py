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
from keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.utils import shuffle
from sklearn.feature_extraction import image

path = r"C:\Users\Mattia\Desktop\Perseverance"


####### CREO UNA LISTA CON ELEMENTI DATI DA QUELLI NELLA CARTELLA DEL PERCORSO ######
dir = os.listdir(path)       #immagini in input

###### INIZIALIZO DUE LISTE, UNA PER LE IMMAGINI E UNA PER LE LABELS ########
image_list = []

#### CICLO FOR PER INSERIRE NELLA LISTA DELLE IMMAGINI IL PERCORSO CORRISPONDENTE ########
for elem in dir:
    new_dir = os.path.join(path,elem)
    if new_dir not in image_list : image_list.append(new_dir)
    
print(image_list[1])
print('Image and label lists dimensions')
print(len(image_list))



crop_images_list=[]
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
for i in range (len(image_list)):
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]
    image = image.astype('float32')
    image/=255                                    
    image = cv2.resize(image, (1024, 1024))
    for r in range (0,8):
        for c in range (0,8):
            cropped_image = image[128*r:128*(r+1),128*c:128*(c+1)]
            cropped_image = cropped_image.astype('float32')
            crop_images_list.append(cropped_image)                                #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo path in image_list

######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
save_cropped_images_TEST(crop_images_list) 

SHAPE=128;
BATCH= 1
x_test = tf.data.Dataset.from_tensor_slices(crop_images_list)
x_test = (
    x_test
    .batch(BATCH)
)

model = tf.keras.models.load_model('model.h5',custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})

predictions = model.predict(x_test,verbose=1,steps=len(crop_images_list))

save_predictions(predictions)