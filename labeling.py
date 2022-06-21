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

print('Image and label lists dimensions')
print(len(image_list))
print(len(label_list))

### DATA AUGMENTATION CON LA FUNZIONE DEFINITA IN UTILS #####
print('[INFO] Data Augmentation...')
tmp1a,tmp2a,A = augment(image_list,label_list);
#A=0                                            ##METTO A=0 COMMENTANDO LA RIGA SOPRA SE NON VOGLIO FARE DATA AUGMENTATION

#N = len(image_list)+A
N=500+A                                      #### UTILIZZARE LA RIGA SOPRA PER USARE TUTTE LE IMMAGINI A DISPOSIZIONE
print('Augmented image list dimension')
print(N)

##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI E DELLE LABEL######
num_classes=5
tmp1 = np.empty((N, 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
tmp2 = np.empty((N, 64, 64, num_classes), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel

print('tmp1, tmp1a, tmp2, tmp2a shapes: ')
print(tmp1.shape)
print(tmp1a.shape)

print(tmp2.shape)
print(tmp2a.shape)

###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
for i in range (N-A):
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    image = cv2.resize(image, (64,64))              #faccio un resize per far combaciare la dimensione dell'input con quello della rete
    image = image.astype('float32')
    image/=510                                      #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    tmp1[i] = image                                 #l'i-esimo elmento di tmp1 sarà dato dall'immagine corrispondente all'i-esimo path in image_list

print('[INFO]Generating image array for augmented data')
for p in range (A):
    print(p)
    image=tmp1a[p]
    # image = image.astype('float32')                 ### LA NORMALIZZAZIONE VIENE GIà FATTA NELLA FUNZIONE utils.augment()
    # image/=510                                      #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    tmp1[N-A+p] = image  

print("[INFO] Images arrays saved")
save_np_arrays(tmp1)                                #salvo tmp1 in un file numpy




### DEFINISCO DEGLI ARRAY RELATIVE ALLE VARIE CLASSI ####
# bedrock=[1/510,1/510,1/510];
# sand=[2/510,2/510,2/510];
# bigrock=[3/510,3/510,3/510];                          ##### HO TOLTO LA NORMALIZZAZIONE PER PROBLEMI CON IL LABELING
# soil=[0,0,0];
# nullo=[255/510,255/510,255/510];

soil=[0,0,0];
bedrock=[1,1,1];
sand=[2,2,2];
bigrock=[3,3,3];
nullo=[255,255,255];

### PER LE LABEL CREO UN ARRAY DI DIMENSIONE 64X64X5 (NEW_LABEL) DOVE 64X64 è LA DIMENSIONE DELL'IMMAGINE
### MENTRE 5 è IL NUMERO DI CLASSI. IN QUESTO MODO HO UN VETTORE DEL TIPO [0 0 1 0 0] PER OGNI PIXEL, CHE INDICA
### A QUALE CLASSE APPARTIENE IL PIXEL (IN QUESTO CASO, ALLA TERZA CLASSE). 
print('[INFO]Generating labels array')
for j in range (N-A):
    print(j)
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]   #leggo l'immagine di label
    label = cv2.resize(label, (64,64))               #ridimension per combaciare con l'input
    # label = label.astype('float32')
    # label/=510                                       #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    # if (j==119):
    #     print(label[:,:])
    reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    new_label = np.empty((64, 64, num_classes), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel

    for t in range(0,num_classes-1):
        new_label[:,:,t]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x5) i valori di reduct_label (64x64)
    

    for i in range(0,63):
        for n in range(0,63): 
            channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la posizione dell'1 nel vettore di dimensione 5
            #print(channels_xy)
            if all(channels_xy==bedrock):      #BEDROCK      
                new_label[i,n,0]=1
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=0
             
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
                
            else: #all(channels_xy==nullo):    #NULL
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=1
    tmp2[j] = new_label
    

print('[INFO]Generating labels array for augmented data')
for f in range (0,A):
    print(f)
    label=tmp2a[f]
    # label = label.astype('float32')
    # label/=510                                       #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    reduct_label=label[:,:,0]                        #definisco una variabile di dimensione 64x64 considerando solo le prime due dimensioni di label
    new_label = np.empty((64, 64, num_classes), dtype=np.uint8)  #inizializzo una nuova lista che andrà a contenere le informazioni per ogni pixel

    for t in range(0,num_classes-1):
        new_label[:,:,t]=reduct_label                  #associo alle prime 2 dimesnioni di new_label (64x64x5) i valori di reduct_label (64x64)
    

    for i in range(0,63):
        for n in range(0,63): 
            channels_xy = label[i,n];           #prendo i valori del pixel [i,j] e li valuto per definire la posizione dell'1 nel vettore di dimensione 5
            #print(channels_xy)
            if all(channels_xy==bedrock):      #BEDROCK      
                new_label[i,n,0]=1
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=0
                
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
                
            else: #all(channels_xy==nullo):    #NULL
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=1
 
    tmp2[N-A+f] = new_label
    

print("[INFO] label arrays saved")
save_np_arrays_labels(tmp2)              #salvo tmp2 in un file numpy

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
