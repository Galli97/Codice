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

###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
# path = r"/content/drive/MyDrive/Tesi/cropped_images.npy"
# path1 = r"/content/drive/MyDrive/Tesi/cropped_labels.npys"

# ####### PERCORSO IN LOCALE #########
path = r"C:\Users\Mattia\Documenti\Github\Codice\cropped_images.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\cropped_labels.npy"

### RECUPERO LE DUE LISTE SALVATE #####
crop_images_list = get_np_arrays(path)          #recupero tmp1 dal file 
crop_labels_list = get_np_arrays(path1)          #recupero tmp2 dal file

### DATA AUGMENTATION CON LA FUNZIONE DEFINITA IN UTILS #####
#tmp1a,tmp2a,A = augment(image_list,label_list);
A=0;
#N = len(crop_images_list)+A
N=1000;
##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI E DELLE LABEL ######
num_classes=5
tmp1 = np.empty((N, 64, 64, 1), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, 64, 64, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe

print('Image and label lists dimensions')
print(len(crop_images_list))
print(len(crop_labels_list))

print('Elem1: ', crop_images_list[0])
print('label1: ', crop_labels_list[0])

####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
crop_images_list, crop_labels_list = shuffle(crop_images_list, crop_labels_list)
print('Elem1 shuffled: ', crop_images_list[0])
print('label1: ', crop_labels_list[0])

print('[INFO]Generating images array')
tmp1 = crop_images_list


soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;

print('[INFO]Generating labels array')
for t in range (0,N-A):
    print(t)
    crop=crop_labels_list[t]
    reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
    #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
    for i in range(0,64):
        for n in range(0,64): 
            channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
            if channels_xy[0]==bedrock:       #BEDROCK      
                new_label[i,n,:]=1
            elif channels_xy[0]==sand:     #SAND
                new_label[i,n,:]=2
            elif channels_xy[0]==bigrock:     #BIG ROCK
                new_label[i,n,:]=3
            elif channels_xy[0]==soil:     #SOIL
                new_label[i,n,:]=0
            elif channels_xy[0]==nullo:    #NULL
                new_label[i,n,:]=4
    tmp2[t] = new_label

print("[INFO] label arrays saved")
save_label_patches(tmp2)

print('tmp1[0]: ', tmp1[0])
print('tmp2[0]: ',tmp2[0])
