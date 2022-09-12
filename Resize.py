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
path = r"C:\Users\Mattia\Desktop\Train_images"
path1 =  r"C:\Users\Mattia\Desktop\Train_labels"

####### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
# path = r"/content/drive/MyDrive/Tesi/Dataset/Train_images"
# path1 = r"/content/drive/MyDrive/Tesi/Dataset/Train_labels"

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

print('Elem1: ', image_list[0])
print('label1: ', label_list[0])

####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
image_list, label_list = shuffle(np.array(image_list), np.array(label_list))
print('Elem1 shuffled: ', image_list[0])
print('label1: ', label_list[0])

####NUMERO DI IMMAGINI NEL DATASET + IMMAGINI DOVUTE AL DATA AUGMENTATION ####
#N = len(image_list)           
N=2500                                 #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE

num_classes=5

#### PRINT DI CONTROLLO ####
print('Augmented image list dimension')
print(N)

SHAPE=128
## INITIALIZE SOME VARIABLES
crop_images_list=[]
crop_labels_list=[]
tmp1 = np.empty((N, SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, SHAPE, SHAPE, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe

coeff=8;

soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;

# IMAGE SELECTION PROCESS #per le 64 sto a 1670-numero attuale
print('[INFO]Generating labels array')
for j in range (0,N):
    print(j)
    #Take the image
    image = cv2.imread(image_list[j])[:,:,[2,1,0]]
    image = image.astype('float32')
    image/=510 
    resized_image = cv2.resize(image, (SHAPE, SHAPE))
    #Take the label
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label=np.expand_dims(label, axis=2)
    label = label.astype('float32')
    resized_label = cv2.resize(label, (SHAPE, SHAPE), 0, 0, interpolation = cv2.INTER_NEAREST)
    crop_images_list.append(resized_image)
    new_label = np.empty((SHAPE, SHAPE, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    new_label[:,:,0] = resized_label 
    tmp2[j] = new_label

tmp1 = crop_images_list
######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
save_cropped_images(tmp1) 

######## SALVATAGGIO ####
print("[INFO] Cropped labels arrays saved")
save_cropped_labels(tmp2) 


print(tmp1[0].shape)
print(tmp2[0].shape)
print(tmp1[0])