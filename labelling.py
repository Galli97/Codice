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

####### PERCORSO IN LOCALE #########
#path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
#path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"

####### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
path = r"/content/drive/MyDrive/Tesi/Dataset/Train_images"
path1 = r"/content/drive/MyDrive/Tesi/Dataset/Train_labels"

####### CREO UNA LISTA CON ELEMENTI DATI DA QUELLI NELLA CARTELLA DEL PERCORSO ######
dir = os.listdir(path)       #immagini in input
dir1 = os.listdir(path1)     #labels date dalle maschere

###### INIZIALIZO DUE LISTE, UNA PER LE IMMAGINI UNA PER LE LABELS ########
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


##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI ######
N = len(image_list)
print(N)
num_classes=5
# tmp1 = np.empty((N, 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
# tmp2 = np.empty((N, 64, 64, 5), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel
tmp1 = np.empty((N, 1024, 1024, 3), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, 1024, 1024, 5), dtype=np.uint8)
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
for i in range (len(image_list)):
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    #image = cv2.resize(image, (64,64))              #faccio un resize per far combaciare la dimensione dell'input con quello della rete
    #print(image.shape)
    tmp1[i] = image                                 #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo pathin image_list
print("[INFO] Images arrays saved")
save_np_arrays(tmp1)                                #salvo tmp1 in un file numpy

### DEFINISCO DEGLI ARRAY RELATIVE ALLE VARIE CLASSI ####
bedrock=[1,1,1];
sand=[2,2,2];
bigrock=[3,3,3];
soil=[255,255,255];
nullo=[0,0,0];

### PER LE LABEL CREO UN ARRAY DI DIMENSIONE 64X64X5 (NEW_LABEL) DOVE 64X64 è LA DIMENSIONE DELL'IMMAGINE
### MENTRE 5 è IL NUMERO DI CLASSI. IN QUESTO MODO HO UN VETTORE DEL TIPO [0 0 1 0 0] PER OGNI PIXEL, CHE INDICA
### A QUALE CLASSE APPARTIENE IL PIXEL (IN QUESTO CASO, ALLA TERZA CLASSE). 
for j in range (len(label_list)):
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]   #leggo l'immagine di label
    #label = cv2.resize(label, (64,64))               #ridimension per combaciare con l'input
    #print(label[0,0])
    reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    #print(reduct_label.shape)
    new_label = np.empty((1024, 1024, 5), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel

    for t in range(0,num_classes-1):
        new_label[:,:,t]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x3) i valori di reduct_label (64x64)

    for i in range(0,1023):
        for n in range(0,1023): 
            channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la posizione dell'1 nel vettore di dimensione 5
            #print(channels_xy)
            if all(channels_xy==bedrock):      #BEDROCK      
                new_label[i,n,0]=1
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=0
                #print(new_label.shape)
            elif all(channels_xy==sand):    #SAND
                new_label[i,n,0]=0
                new_label[i,n,1]=1
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=0
                
            elif all(channels_xy==bigrock):    #BIG ROCK
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=1
                new_label[i,n,3]=0
                new_label[i,n,4]=0
                
            elif all(channels_xy==soil):    #SOIL
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=1
                new_label[i,n,4]=0
                
            elif all(channels_xy==nullo):    #NULL
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=1
    #print(new_label.shape)
    tmp2[j] = new_label
    #print(tmp2.shape)

print("[INFO] label arrays saved")
save_np_arrays_labels(tmp2)              #salvo tmp2 in un file numpy

print('[TODO] Download these two files from the colab folder and save on the drive')


##### ALCUNI PRINT DI CONTROLLO ######
#print(len(image_list))
#print(len(label_list))

#print(image_list)
#print(label_list)

#print(len(tmp1))
#print(tmp2)
#print(tmp2[1,1,1])
#print(new_label[:,:,1])
