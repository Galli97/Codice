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
####### PERCORSO IN LOCALE #########

path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-images"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-labels"

####### CREO UNA LISTA CON ELEMENTI DATI DA QUELLI NELLA CARTELLA DEL PERCORSO ######
dir = os.listdir(path)       #immagini in input
dir1 = os.listdir(path1)     #labels date dalle maschere

###### INIZIALIZO DUE LISTE, UNA PER LE IMMAGINI E UNA PER LE LABELS ########
image_list = []
label_list = []

#### CICLO FOR PER INSERIRE NELLA LISTA DELLE IMMAGINI IL PERCORSO CORRISPONDENTE ########
for elem in dir:
    new_dir = os.path.join(path,elem)
    if new_dir not in image_list : image_list.append(new_dir)
    #image=np.expand_dims(image, axis=2)
    
#### CICLO FOR PER INSERIRE NELLA LISTA DELLE LABELS IL PERCORSO CORRISPONDENTE ########
for lab in dir1:
    new_dir1 = os.path.join(path1,lab)
    if new_dir1 not in label_list : label_list.append(new_dir1)
    #label=np.expand_dims(label, axis=2)

print('Image and label lists dimensions')
print(len(image_list))
print(len(label_list))

SHAPE=512;
crop_images_list=[]
crop_labels_list=[]
print('[INFO]Generating labels array')
for j in range (0,len(image_list)):
    print(j)
    #Take the image
    image = cv2.imread(image_list[j])[:,:,[2,1,0]]
    image = image.astype('float32')
    image/=255 
    resized_image = cv2.resize(image, (SHAPE, SHAPE))
    crop_images_list.append(resized_image)     
    #Take the label
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label=np.expand_dims(label, axis=2)
    label = label.astype('float32')
    resized_label = cv2.resize(label, (SHAPE, SHAPE), 0, 0, interpolation = cv2.INTER_NEAREST)
    new_label = np.empty((SHAPE, SHAPE, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    new_label[:,:,0] = resized_label 
    crop_labels_list.append(new_label)
   

######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
save_cropped_images_TEST(crop_images_list) 
print('shape ', crop_images_list[0].shape)

######## SALVATAGGIO ####
print("[INFO] Cropped labels arrays saved")
print('shape ', crop_labels_list[0].shape)
save_cropped_labels_TEST(crop_labels_list) 