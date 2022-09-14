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
# path = r"/content/drive/MyDrive/Tesi/cropped_images_TEST.npy"
# path1 = r"/content/drive/MyDrive/Tesi/cropped_labels_TEST.npys"

# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Desktop\Tentativi128\cropped_images_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128\cropped_labels_TEST.npy"

path = r"C:\Users\Mattia\Documenti\github\Codice\cropped_images_TEST.npy"
path1 =  r"C:\Users\Mattia\Documenti\github\Codice\cropped_labels_TEST.npy"

# path = r"C:\Users\Mattia\Desktop\cropped_images_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\cropped_labels_TEST.npy"

### RECUPERO LE DUE LISTE SALVATE #####
crop_images_list = get_np_arrays(path)          #recupero tmp1 dal file 
crop_labels_list = get_np_arrays(path1)          #recupero tmp2 dal file

N = len(crop_images_list)

##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI E DELLE LABEL ######
num_classes=5
# tmp1 = np.empty((len(crop_images_list), 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, 128, 128, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe
tmp3 = np.empty((N, 128, 128, 3), dtype=np.uint8)  #Qui ho N immagini
print('Image and label lists dimensions')
print(len(crop_images_list))
print(len(crop_labels_list))

print('Elem1: ', crop_images_list[0])
print('label1: ', crop_labels_list[0])

# ####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
# crop_images_list, crop_labels_list = shuffle(crop_images_list, crop_labels_list)
# print('Elem1 shuffled: ', crop_images_list[0])
# print('label1: ', crop_labels_list[0])

print('[INFO]Generating images array')
#tmp1 = crop_images_list

#print(tmp1.shape)
img_labelled=[];
label_labelled=[];

soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;

count=0;

chosen_label=[];
print('[INFO]Generating labels array')
for t in range (0,len(crop_labels_list)):
    print('Label: ', t)
    crop=crop_labels_list[t]
    for i in range(0,128):
        for n in range(0,128): 
            channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
            if channels_xy==bedrock:       #BEDROCK      
                crop[i,n,:]=1
            elif channels_xy==sand:     #SAND
                crop[i,n,:]=2
            elif channels_xy==bigrock:     #BIG ROCK
                crop[i,n,:]=3
            elif channels_xy==soil:     #SOIL
                crop[i,n,:]=4
            elif channels_xy==nullo:    #NULL
                crop[i,n,:]=0
    tmp2[count] = crop
    chosen_label.append(t)
    count+=1
    print('count: ', count)
print('chosen_label',len(chosen_label))

selected_images=[]
for k in range (len(chosen_label)):
    index = chosen_label[k]
    selected_images.append(crop_images_list[index])
print('selected_images',len(selected_images))

print("[INFO] image arrays saved")
tmp3=np.array(selected_images)
save_patches_TEST(tmp3)
print("[INFO] label arrays saved")
save_label_patches_TEST(tmp2)
print('tmp3: ', tmp3.shape)
print('tmp2: ', tmp2.shape)
print('tmp3[0]: ', tmp3[0])
print('tmp2[0]: ',tmp2[0])
