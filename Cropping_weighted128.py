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
N = len(image_list)           
#N=163                                 #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE
print('Augmented image list dimension')
print(N)

num_classes=5

#### PRINT DI CONTROLLO ####
print('Augmented image list dimension')
print(N)


crop_images_list=[]
crop_labels_list=[]

soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;

flag_sand=False;
flag_bedrock=False;
flag_bigrock=False;
flag_soil=False;
flag_null=False;
counter_sand=0;
counter_bedrock=0;
counter_bigrock=0;
counter_soil=0;
counter_null=0;
counter_soil_reduce=0;
count=0;

print('[INFO]Generating labels array')
for j in range (0,N):
    if(count==3000):
        break
    flag_sand=False;
    flag_bedrock=False;
    flag_bigrock=False;
    flag_soil=False;
    flag_null=False;
    flag_selected=False;
    counter_sand=0;
    counter_bedrock=0;
    counter_bigrock=0;
    counter_soil=0;
    counter_null=0;
    counter_soil_reduce=0;
    print(j)
    image = cv2.imread(image_list[j])[:,:,[2,1,0]]
    image = image.astype('float32')
    image/=510 
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = np.expand_dims(label, axis=2)
    label = label.astype('float32')
    for r in range (0,8):
        flag_selected=False;
        for k in range (0,112):  #113 !!!
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            flag_null=False;
            counter_sand=0;
            counter_bedrock=0;
            counter_bigrock=0;
            counter_soil=0;
            counter_null=0;
            counter_soil_reduce=0;
        
            if(flag_selected==True):
                break

            cropped_label = label[128*(r):128*(r+1),8*k:128+8*k]           #Passo 5
            cropped_image = image[128*(r):128*(r+1),8*k:128+8*k]
            for m in range(0,128):
                if(flag_null==True):
                    break
                # elif(flag_bigrock==True and flag_sand==True):
                #     break
                # elif(flag_bigrock==True and flag_bedrock==True):
                #     break
                # elif(flag_bigrock==True and flag_soil==True):
                #     break
                # elif(flag_bedrock==True and flag_soil==True):
                #     break
                # elif(flag_bedrock==True and flag_sand==True):
                #     break
                # elif(flag_sand==True and flag_soil==True):
                #     break
                for s in range(0,128): 
                    channels_xy = cropped_label[m,s];
                    if(flag_null==True):
                        break
                    # elif(flag_bigrock==True and flag_sand==True):
                    #     break
                    # elif(flag_bigrock==True and flag_soil==True):
                    #     break
                    # elif(flag_bigrock==True and flag_bedrock==True):
                    #     break
                    # elif(flag_bedrock==True and flag_soil==True):
                    #     break
                    # elif(flag_bedrock==True and flag_sand==True):
                    #     break
                    # elif(flag_sand==True and flag_soil==True):
                    #     break
                    elif channels_xy==nullo:    #Null
                        counter_null+=1;
                        if (counter_null>8192):
                            flag_null=True;
                    elif channels_xy==bedrock:      #BEDROCK
                        counter_bedrock+=1;
                        if (counter_bedrock>4096):
                            flag_bedrock=True
                    elif channels_xy==sand:    #SAND
                        counter_sand+=1;
                        if (counter_sand>4096):
                            flag_sand=True
                    elif channels_xy==bigrock:    #BIG ROCK
                        counter_bigrock+=1;
                        if (counter_bigrock>4096):
                            flag_bigrock=True
                    elif channels_xy==soil:    #SOIL
                        counter_soil+=1;
                        if (counter_soil>4096):
                            counter_soil_reduce+=1;
                            flag_soil=True;    
            if(flag_null==True):
                continue

            if(flag_soil==True):
                if (counter_soil_reduce>2048):
                    flag_soil=False;  
                else:
                    flag_soil=True;
            
            if (flag_bigrock==True and flag_sand==True):
                print('Big Rock-sand IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)
            elif (flag_bigrock==True and flag_bedrock==True):
                print('Big Rock-bedrock IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)
            elif (flag_bigrock==True and flag_soil==True):
                print('Big Rock-soil IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)
            elif (flag_bedrock==True and flag_sand==True):
                print('BedRock-Sand IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)
            elif (flag_bedrock==True and flag_soil==True):
                print('BedRock-Soil IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)
            elif (flag_sand==True and flag_soil==True):
                print('Sand-Soil IN')
                crop_labels_list.append(cropped_label)
                crop_images_list.append(cropped_image)
                flag_selected=True;
                flag_sand=False;
                flag_bedrock=False;
                flag_bigrock=False;
                flag_soil=False;
                flag_null=False;
                count+=1
                print('count: ', count)

######## SALVATAGGIO ####
print("[INFO] Cropped images arrays saved")
save_cropped_images(crop_images_list) 

######## SALVATAGGIO ####
print("[INFO] Cropped labels arrays saved")
save_cropped_labels(crop_labels_list) 


print(crop_images_list[0].shape)
print(crop_labels_list[0].shape)
print(crop_labels_list[0])