## Here we take the selected labels and we define the classes of each pixel
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
SHAPE = 512;
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

flag_sand=False;
flag_bedrock=False;
flag_bigrock=False;
flag_soil=False;
count=0;

chosen_label=[];
print('[INFO]Generating labels array')
for t in range (0,len(crop_labels_list)):
    print('Label: ', t)
    flag_sand=False;
    flag_bedrock=False;
    flag_bigrock=False;
    flag_soil=False;
    if(count==N):
        break
    # else:
    #     image = crop_labels_list[t]
    #     for m in range(0,64):
    #         for j in range(0,64): 
    #             channels_xy = image[m,j];
    #             if channels_xy[0]==bedrock:       #BEDROCK      
    #                 image[m,j,:]=1
    #             elif channels_xy[0]==sand:     #SAND
    #                 image[m,j,:]=2
    #             elif channels_xy[0]==bigrock:     #BIG ROCK
    #                 image[m,j,:]=3
    #             elif channels_xy[0]==soil:     #SOIL
    #                 image[m,j,:]=4
    #             elif channels_xy[0]==nullo:    #NULL
    #                 image[m,j,:]=0
    #     tmp2[count] = image
    #     chosen_label.append(t) #(tmp1[t])
    #     count+=1
    else:
        image = crop_labels_list[t]
        #print(image.shape)
        for m in range(0,SHAPE):
            if(flag_bigrock==True and flag_soil==True):
                break
            elif(flag_bigrock==True and flag_sand==True):
                break
            elif(flag_bigrock==True and flag_bedrock==True):
                break
            elif(flag_bedrock==True and flag_soil==True):
                break
            elif(flag_bedrock==True and flag_sand==True):
                break
            elif(flag_sand==True and flag_soil==True):
                break
            for j in range(0,SHAPE): 
                channels_xy = image[m,j];
                if(flag_bigrock==True and flag_soil==True):
                    break
                elif(flag_bigrock==True and flag_sand==True):
                    break
                elif(flag_bigrock==True and flag_bedrock==True):
                    break
                elif(flag_bedrock==True and flag_soil==True):
                    break
                elif(flag_bedrock==True and flag_sand==True):
                    break
                elif(flag_sand==True and flag_soil==True):
                    break
                elif channels_xy==bedrock:      #BEDROCK
                    flag_bedrock=True
                elif channels_xy==sand:    #SAND
                    flag_sand=True
                elif channels_xy==bigrock:    #BIG ROCK
                    flag_bigrock=True
                elif channels_xy==soil:    #BIG ROCK
                    flag_soil=True
        if (flag_bigrock==True and flag_sand==True):
            print('Big Rock-sand IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t) #(tmp1[t])
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
        elif (flag_bigrock==True and flag_bedrock==True):
            print('Big Rock-bedrock IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
        elif (flag_bigrock==True and flag_soil==True):
            print('Big Rock-soil IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
        elif (flag_bedrock==True and flag_sand==True):
            print('BedRock-Sand IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
        elif (flag_bedrock==True and flag_soil==True):
            print('BedRock-Soil IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
        elif (flag_sand==True and flag_soil==True):
            print('Sand-Soil IN')
            print('Inserted label: ',t)
            crop=crop_labels_list[t]
            # reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
            # new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
            # new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
            #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
            for i in range(0,SHAPE):
                for n in range(0,SHAPE): 
                    channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
                    if channels_xy[0]==bedrock:       #BEDROCK      
                        crop[i,n,:]=1
                    elif channels_xy[0]==sand:     #SAND
                        crop[i,n,:]=2
                    elif channels_xy[0]==bigrock:     #BIG ROCK
                        crop[i,n,:]=3
                    elif channels_xy[0]==soil:     #SOIL
                        crop[i,n,:]=4
                    elif channels_xy[0]==nullo:    #NULL
                        crop[i,n,:]=0
            tmp2[count] = crop
            chosen_label.append(t)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print('count: ', count)
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