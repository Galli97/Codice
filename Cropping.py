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

### DATA AUGMENTATION CON LA FUNZIONE DEFINITA IN UTILS #####
#tmp1a,tmp2a,A = augment(image_list,label_list);
#A=0;                                              #### METTO A=0 SE NON VOGLIO FARE DATA AUGMENTATION, COMMENTANDO LA RIGA SOPRA

####NUMERO DI IMMAGINI NEL DATASET + IMMAGINI DOVUTE AL DATA AUGMENTATION ####
N = 1 #len(image_list)           
#N=163                                 #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE
print('Augmented image list dimension')
print(N)

num_classes=5

#### PRINT DI CONTROLLO ####
print('Augmented image list dimension')
print(N)


crop_images_list=[]
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
for i in range (N):
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image=np.expand_dims(image, axis=2)
    image = image.astype('float32')
    image/=255                                  #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    for r in range (0,16):
        for c in range (0,16):
            cropped_image = image[64*r:64*(r+1),64*c:64*(c+1)]
            crop_images_list.append(cropped_image)                                #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo path in image_list

######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
save_cropped_images(crop_images_list) 

crop_labels_list=[]

print('[INFO]Generating labels array')
for j in range (N):
    print(j)
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # label=np.expand_dims(label, axis=2)
    label = label.astype('float32')
    for r in range (0,16):
        for c in range (0,16):
            cropped_label = label[64*(r):64*(r+1),64*(c):64*(c+1)]
            #cropped_label = cropped_label.astype('float32')
            crop_labels_list.append(cropped_label)

######## SALVATAGGIO ####
print("[INFO] Cropped labels arrays saved")
save_cropped_labels(crop_labels_list) 


print(crop_images_list[0].shape)
print(crop_labels_list[0].shape)
print(crop_labels_list[5])