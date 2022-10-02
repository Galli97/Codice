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
    image = cv2.resize(image, (128, 128))
    crop_images_list.append(image)                                #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo path in image_list

######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
#save_cropped_images_TEST(crop_images_list) 
tmp1 = crop_images_list
SHAPE=128;
BATCH= 1
x_test = tf.data.Dataset.from_tensor_slices(crop_images_list)
x_test = (
    x_test
    .batch(BATCH)
)

model = tf.keras.models.load_model('model.h5',custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})

predictions = model.predict(x_test,verbose=1,steps=len(crop_images_list))

#save_predictions(predictions)
SHAPE=128;


i = 1 #random.randint(0,9)
print(i)

predictions = decode_predictions(predictions,SHAPE)
prediction = decode_masks(predictions,SHAPE)

resized = tmp1[i]
print(resized.shape)
predictions = prediction[i]
print(predictions.shape)
resized = np.asarray(resized, np.float32)
predictions = np.asarray(predictions, np.float32)
overlay_img = cv2.addWeighted(resized, 0.9, predictions,  0.001, 0)
result = cv2.resize(overlay_img, (512, 512))
cv2.imshow('total_pred',result)
cv2.waitKey(0)