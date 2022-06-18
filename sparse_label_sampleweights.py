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

print('image and label lists dimensions')
print(len(image_list))
print(len(label_list))

### DATA AUGMENTATION CON LA FUNZIONE DEFINITA IN UTILS #####
#image_list_aug, label_list_aug = augment(image_list,label_list);
#tmp1a,tmp2a,A = augment(image_list,label_list);
A=0;

##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI ######
N = len(image_list)+A
print('Augmented image list dimension')
print(N)
num_classes=5
tmp1 = np.empty((N, 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
#tmp2 = np.empty((N, 64, 64, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel
tmp2 = np.empty((N, 4096, 1), dtype=np.uint8)

print('tmp1,tmp1a,tmp2,tmp2a shapes')
print(tmp1.shape)
#print(tmp1a.shape)

print(tmp2.shape)
#print(tmp2a.shape)
print('Number of augmented images')
print(A)
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
for i in range (N-A):
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    image = cv2.resize(image, (64,64))              #faccio un resize per far combaciare la dimensione dell'input con quello della rete
    image = image.astype('float32')
    image/=510                                      #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    #print(image.shape)
    tmp1[i] = image                                 #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo pathin image_list

# for p in range (A):
#     print(p)
#     image=tmp1a[p]
#     image = image.astype('float32')
#     image/=510                                      #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
#     #print(image.shape)
#     tmp1[N-A+p] = image  

print("[INFO] Images arrays saved")
save_sparse_np_arrays(tmp1)                                #salvo tmp1 in un file numpy




### DEFINISCO DEGLI ARRAY RELATIVE ALLE VARIE CLASSI ####
bedrock=[1/510,1/510,1/510];
sand=[2/510,2/510,2/510];
bigrock=[3/510,3/510,3/510];
soil=[255/510,255/510,255/510];
nullo=[0,0,0];

### PER LE LABEL CREO UN ARRAY DI DIMENSIONE 64X64X5 (NEW_LABEL) DOVE 64X64 è LA DIMENSIONE DELL'IMMAGINE
### MENTRE 5 è IL NUMERO DI CLASSI. IN QUESTO MODO HO UN VETTORE DEL TIPO [0 0 1 0 0] PER OGNI PIXEL, CHE INDICA
### A QUALE CLASSE APPARTIENE IL PIXEL (IN QUESTO CASO, ALLA TERZA CLASSE). 
for j in range (N-A):
    print(j)
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]   #leggo l'immagine di label
    label = cv2.resize(label, (64,64))               #ridimension per combaciare con l'input
    label = label.astype('float32')
    label/=510                                       #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    #print(label[0,0])
    reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    #print(reduct_label.shape)
    new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
    new_lab = np.empty((4096, 1), dtype=np.uint8)
    new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x5) i valori di reduct_label (64x64)

    for i in range(0,63):
        for n in range(0,63): 
            channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la classe di appartenenza del pixel
            #print(channels_xy)
            if all(channels_xy==bedrock):      #BEDROCK      
                new_label[i,n,0]=0
            elif all(channels_xy==sand):    #SAND
                new_label[i,n,0]=1
            elif all(channels_xy==bigrock):    #BIG ROCK
                new_label[i,n,0]=2
            elif all(channels_xy==soil):    #SOIL
                new_label[i,n,0]=3
            elif all(channels_xy==nullo):    #NULL
                new_label[i,n,0]=4
    #print(new_label.shape)
    new_label = cv2.resize(new_label, (4096,1)) 
    new_lab[:,0] = new_label                    
    tmp2[j] = new_lab                           
    #print(tmp2.shape)

# for f in range (0,A):
#     print(f)
#     label=tmp2a[f]
#     label = label.astype('float32')
#     label/=510                                       #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
#     #print(label[0,0])
#     reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
#     #print(reduct_label.shape)
#     new_label = np.empty((64, 64, 1), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel
#     new_label[:,:,0]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x5) i valori di reduct_label (64x64)
#     for i in range(0,63):
#         for n in range(0,63): 
#             channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la posizione dell'1 nel vettore di dimensione 5
#             #print(channels_xy)
#             if all(channels_xy==bedrock):      #BEDROCK      
#                 new_label[i,n,0]=0
#             elif all(channels_xy==sand):    #SAND
#                 new_label[i,n,0]=1
#             elif all(channels_xy==bigrock):    #BIG ROCK
#                 new_label[i,n,0]=2
#             elif all(channels_xy==soil):    #SOIL
#                 new_label[i,n,0]=3
#             elif all(channels_xy==nullo):    #NULL
#                 new_label[i,n,0]=4
#     #print(new_label.shape)
#     tmp2[N-A+f] = new_label
 
#print('tmp2[0]')
#print(tmp2[0])
print("[INFO] label arrays saved")
save_sparse_np_arrays_labels(tmp2)              #salvo tmp2 in un file numpy

print('[TODO] Download these two files from the colab folder and save on the drive')


##### ALCUNI PRINT DI CONTROLLO ######
#print(len(image_list))
#print(len(label_list))

#print(image_list)
#print(label_list)

#print(len(tmp1))
#print(len(tmp2))

#print(tmp1.shape)
#print(tmp2.shape)
#print(tmp2)
#print(tmp2[1,1,1])
#print(new_label[:,:,1])