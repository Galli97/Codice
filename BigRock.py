
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
# image_list, label_list = shuffle(np.array(image_list), np.array(label_list))
# print('Elem1 shuffled: ', image_list[0])
# print('label1: ', label_list[0])

####NUMERO DI IMMAGINI NEL DATASET + IMMAGINI DOVUTE AL DATA AUGMENTATION ####
#N = len(image_list)           
N=5                                 #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE

num_classes=5

#### PRINT DI CONTROLLO ####
print('Augmented image list dimension')
print(N)

## INITIALIZE SOME VARIABLES
crop_images_list=[]
crop_labels_list=[]

SHAPE=128;
coeff=8;

soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;


flag_bigrock=False;
flag_bedrock=False;
flag_sand=False;
flag_soil=False;

counter_bigrock=0;
counter_bedrock=0;
counter_sand=0;
counter_soil=0;
count=0;



# IMAGE SELECTION PROCESS #per le 64 sto a 1670-numero attuale
print('[INFO]Generating labels array')
for j in range (0,1000):

    if(count==1500):
        break

    flag_bigrock=False;
    take_bigrock=False;
    flag_selected=False;
    flag_soil=False;
    flag_sand=False;
    flag_bedrock=False;
    counter_bigrock=0;
    counter_bedrock=0;
    counter_sand=0;
    counter_soil=0;
   
    print(j)

    #Take the image
    image = cv2.imread(image_list[j])[:,:,[2,1,0]]
    image = image.astype('float32')
    image/=510 
    #Take the label
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label=np.expand_dims(label, axis=2)
    label = label.astype('float32')
    #Start the process
    for r in range (0,1024):
        if(flag_bigrock==True):
            break
        for c in range (0,1024):
            channels_xy = label[r,c]
            if(channels_xy==bigrock):
                flag_bigrock=True;
                break
    if (flag_bigrock==False):
        continue

    elif(flag_bigrock==True):
        for v in range (0,int(1024/SHAPE)):
            flag_selected=False;
            for k in range (0,int((1024-SHAPE)/coeff)):
                counter_bigrock=0;
                counter_bedrock=0;
                counter_sand=0;
                counter_soil=0;
                if(flag_selected==True):
                    break
                cropped_label = label[SHAPE*(v):SHAPE*(v+1),coeff*k:SHAPE+coeff*k]           #Passo coeff
                for m in range(0,SHAPE):
                    if(take_bigrock==True):
                        break
                    for s in range(0,SHAPE):
                        channels = cropped_label[m,s];
                        if channels==bigrock:    #BIG ROCK
                            counter_bigrock+=1;
                            if (counter_bigrock>8000):
                                take_bigrock=True;
                                break
                        if channels==bedrock:
                            counter_bedrock+=1;
                            if (counter_bedrock>500):
                                flag_bedrock=True;
                        if channels==sand:    
                            counter_sand+=1;
                            if (counter_sand>500):
                                flag_sand=True;
                        if channels==soil:    
                            counter_soil+=1;
                            if (counter_soil>500):
                                flag_bedrock=True;
                        else:
                            continue
                if (take_bigrock==True and flag_soil==True or take_bigrock==True and flag_sand==True or take_bigrock==True and flag_bedrock==True):
                    print('Big Rock IN')
                    cropped_image = image[SHAPE*(v):SHAPE*(v+1),coeff*k:SHAPE+coeff*k]
                    crop_labels_list.append(cropped_label)
                    crop_images_list.append(cropped_image)
                    print('flag_bedrock:',flag_bedrock)
                    print('flag_bigrock:',flag_bigrock)
                    print('flag_sand:',flag_sand)
                    print('flag_soil:',flag_soil)
                    flag_selected=True;
                    take_bigrock=False;
                    flag_soil=False;
                    flag_sand=False;
                    flag_bedrock=False;
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