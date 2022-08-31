##Here we apply a fucntion to do some data augmentation
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

# path = r"/content/drive/MyDrive/Tesi/image_patches.npy"
# path1 = r"/content/drive/MyDrive/Tesi/label_patches.npy"

# ####### PERCORSO IN LOCALE #########
path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"

tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print(tmp1.shape)
print('0: ',tmp1[0])
tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)
print('0: ',tmp2[0])

N=len(tmp1)
SHAPE=128;
### DATA AUGMENTATION CON LA FUNZIONE DEFINITA IN UTILS #####
tmp1a,tmp2a,A = augment(tmp1,tmp2,N);
# print('A: ',A)

# print(tmp1a[1])
# augmented_images = np.empty((A, 64, 64, 3), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe
# augmented_labels= np.empty((A, 64, 64, 1), dtype=np.uint8)  #Qui ho N immagini

# print('[INFO]Generating images array for augmented data')
# images=[]
# for p in range (A):
#     print(p)
#     image=tmp1a[p]
#     #print(image)
#     images.append(image)  
# augmented_images=np.array(images)
# print(augmented_images.shape)
# print(augmented_images[0])
# print(augmented_images[A-1])
# soil=0;
# bedrock=1;
# sand=2;
# bigrock=3;
# nullo=255;
# print('[INFO]Generating labels array for augmented data')
# for f in range (0,A):
#     print(f)
#     label=tmp2a[f]
    # reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x5) i valori di reduct_label (64x64)
    # for i in range(0,64):
    #     for n in range(0,64): 
    #         channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
    #         if channels_xy[0]==bedrock:       #BEDROCK      
    #             new_label[i,n,:]=1
    #         elif channels_xy[0]==sand:     #SAND
    #             new_label[i,n,:]=2
    #         elif channels_xy[0]==bigrock:     #BIG ROCK
    #             new_label[i,n,:]=3
    #         elif channels_xy[0]==soil:     #SOIL
    #             new_label[i,n,:]=4
    #         elif channels_xy[0]==nullo:    #NULL
    #             new_label[i,n,:]=0
    # augmented_labels[f] = new_label

# print(augmented_labels.shape)
# print(augmented_labels[0])
# print(augmented_labels[A-1])

image_dataset=np.concatenate((tmp1,tmp1a))
label_dataset=np.concatenate((tmp2,tmp2a))

save_final_images(image_dataset)
save_final_labels(label_dataset)

print(tmp1a[0])
print(tmp2a[0])

print(image_dataset.shape)
print(label_dataset.shape)

masks=decode_masks_Notsparse(tmp2a[0:3],SHAPE)
label1=cv2.resize(masks[0],(512,512))
label2=cv2.resize(masks[1],(512,512))
label3=cv2.resize(masks[2],(512,512))
cv2.imshow('image', label1)
cv2.waitKey(0) 
cv2.imshow('image', label2)
cv2.waitKey(0) 
cv2.imshow('image', label3)
cv2.waitKey(0)  

# print(tmp2[0])
# print(label_dataset[0])