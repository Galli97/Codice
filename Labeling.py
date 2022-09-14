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


#N = len(crop_images_list)+A
N=8000;
SHAPE = 128;
##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI E DELLE LABEL ######
num_classes=5
tmp1 = np.empty((len(crop_images_list), SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, SHAPE, SHAPE, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe
tmp3 = np.empty((N, SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
print('Image and label lists dimensions')
print(len(crop_images_list))
print(len(crop_labels_list))

# print('Elem1: ', crop_images_list[0])
# print('label1: ', crop_labels_list[0])

# ####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
# crop_images_list, crop_labels_list = shuffle(crop_images_list, crop_labels_list)
# print('Elem1 shuffled: ', crop_images_list[0])
# print('label1: ', crop_labels_list[0])

print('[INFO]Generating images array')
tmp1 = crop_images_list


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
    if(count==N):
        break
    else:
        image = crop_labels_list[t]
        for m in range(0,SHAPE):
            for j in range(0,SHAPE): 
                channels_xy = image[m,j];
                if channels_xy[0]==bedrock:       #BEDROCK      
                    image[m,j,:]=1
                elif channels_xy[0]==sand:     #SAND
                    image[m,j,:]=2
                elif channels_xy[0]==bigrock:     #BIG ROCK
                    image[m,j,:]=3
                elif channels_xy[0]==soil:     #SOIL
                    image[m,j,:]=4
                elif channels_xy[0]==nullo:    #NULL
                    image[m,j,:]=0
        tmp2[count] = image
        chosen_label.append(t) #(tmp1[t])
        count+=1
print('chosen_label',len(chosen_label))
selected_images=[]
for k in range (len(chosen_label)):
    index = chosen_label[k]
    selected_images.append(tmp1[index])

print('selected_images',len(selected_images))
print("[INFO] image arrays saved")
tmp3=np.array(selected_images)
save_patches(tmp3)
print("[INFO] label arrays saved")
tmp2=tmp2[0:len(tmp3)]
save_label_patches(tmp2)
print('tmp3: ', tmp3.shape)
print('tmp2: ', tmp2.shape)
print('tmp3[0]: ', tmp3[0])
print('tmp2[0]: ',tmp2[0])
masks=decode_masks(tmp2,SHAPE)
label1=cv2.resize(masks[0],(512,512))
label2=cv2.resize(masks[1],(512,512))
label3=cv2.resize(masks[2],(512,512))
cv2.imshow('image', label1)
cv2.waitKey(0) 
cv2.imshow('image', label2)
cv2.waitKey(0) 
cv2.imshow('image', label3)
cv2.waitKey(0)  