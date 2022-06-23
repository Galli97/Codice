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
A=0;                                              #### METTO A=0 SE NON VOGLIO FARE DATA AUGMENTATION, COMMENTANDO LA RIGA SOPRA

####NUMERO DI IMMAGINI NEL DATASET + IMMAGINI DOVUTE AL DATA AUGMENTATION ####
#N = len(image_list)+A           
N=10                                   #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE
print('Augmented image list dimension')
print(N)

##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI E DELLE LABEL ######
num_classes=5
tmp1 = np.empty((N*256, 64, 64, 1), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N*256, 64, 64, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe

#### PRINT DI CONTROLLO ####
print('Augmented image list dimension')
print(N)

print('tmp1,tmp1a,tmp2,tmp2a shapes: ')

print(tmp1.shape)
#print(tmp1a.shape)

print(tmp2.shape)
#print(tmp2a.shape)

print('Number of augmented images')
print(A)


###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
for i in range (N-A):
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=np.expand_dims(image, axis=2)
    #print(image.shape)
    # if i==0:
    #    cv2.imshow('image',image)
    #    cv2.waitKey(0) 
    image = image.astype('float32')
    image/=510                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    # cropped_image = image[0:64,0:64]
    # tmp1[i] = cropped_image 
    for r in range (0,16):
        for c in range (0,16):
            cropped_image = image[64*r:64*(r+1),64*c:64*(c+1)]
            # print('cropped_image: ',cropped_image.shape)
            # if r==5 and c==3:
            #     cv2.imshow('cropped',cropped_image)
            #     cv2.waitKey(0) 
            tmp1[i] = cropped_image                                 #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo path in image_list

######## SALVATAGGIO ####
print("[INFO] Images arrays saved")
save_patches(tmp1) 


soil=0;
bedrock=1;
sand=2;
bigrock=3;
nullo=255;

cropped_list=[]

### PER LE LABEL CREO UN ARRAY DI DIMENSIONE 64X64X1 (NEW_LABEL) DOVE 64X64 è LA DIMENSIONE DELL'IMMAGINE
### MENTRE L'ULTIMA DIMESIONE CONTIENE UN INTERO DA 0 A 3 CHE RAPPRESENTA A QUALE CLASSE APPARTIENE IL PIXEL CORRISPONDENTE.
print('[INFO]Generating labels array')
for j in range (N-A):
    print(j)
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]   #leggo l'immagine di label
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # label=np.expand_dims(label, axis=2)
    #print(label)
    label = label.astype('float32')
    # cropped_label = label[0:64,0:64]
    # cropped_list.append(cropped_label)
    for r in range (0,16):
        for c in range (0,16):
            #print('patch: ', r,c)
            cropped_label = label[64*(r):64*(r+1),64*(c):64*(c+1)]
            cropped_label = cropped_label.astype('float32')
            cropped_list.append(cropped_label)
            # if r==0 and c==0:
            #     print('cropped label: ',cropped_label)
for t in range (len(cropped_list)):
    print(t)
    crop=cropped_list[t]
    # if t==0:
    #     print(crop)
    #print('crop shape: ', crop.shape)
    reduct_label = crop[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    #print('reduct label: ', reduct_label.shape)
    # if r==0 and c==0:
    #     print('reduct label: ', reduct_label)
    new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x1) i valori di reduct_label (64x64)
    # if r==0 and c==0:
    #     print('new_label: ', new_label)
    #### CONTROLLO OGNI PIXEL PER ASSEGNARE LA CLASSE #######
    for i in range(0,64):
        for n in range(0,64): 
            channels_xy = crop[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
            #if r==0 and c==0:
            #print('channel_xy: ',channels_xy)
            if channels_xy[0]==bedrock:       #BEDROCK      
                new_label[i,n,:]=1
                #print('bed rock: ',channels_xy)
            elif channels_xy[0]==sand:     #SAND
                new_label[i,n,:]=2
                #print('sand: ',channels_xy)
            elif channels_xy[0]==bigrock:     #BIG ROCK
                new_label[i,n,:]=3
                #print('big rock: ',channels_xy)
            elif channels_xy[0]==soil:     #SOIL
                new_label[i,n,:]=4
                #print('soil: ',channels_xy)
            elif channels_xy[0]==nullo:    #NULL
                new_label[i,n,:]=5
                #print('tana')
            #     print(channels_xy)
            #     print(j) 
    tmp2[t] = new_label

print("[INFO] label arrays saved")
save_label_patches(tmp2)

# label = cv2.imread(label_list[100])[:,:,[2,1,0]]   #leggo l'immagine di label
# label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
# label=np.expand_dims(label, axis=2)
# print(label[:,:])
# print(tmp1.shape)
# print(tmp2.shape)
