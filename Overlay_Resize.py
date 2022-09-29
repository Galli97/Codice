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


# path = r"C:\Users\Mattia\Desktop\Tentativi128\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128\label_patches_TEST.npy"
# path2 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions.npy"

# tmp1 = get_np_arrays(path)
# tmp2 = get_np_arrays(path1)
# predictions = get_np_arrays(path2)

# tmp1=tmp1[:64]
# tmp2=tmp2[:64]
# predictions=predictions[:64]
####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-images"
# path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-labels"

path = r"C:\Users\Mattia\Desktop\TEST_images"
path1 =  r"C:\Users\Mattia\Desktop\TEST_labels"

path2 = r"C:\Users\Mattia\Desktop\Resized_Test\Gold\image_patches_TEST.npy"
path3 =  r"C:\Users\Mattia\Desktop\Resized_Test\Gold\label_patches_TEST.npy"
path4 = r"C:\Users\Mattia\Documenti\Github\Codice\predictions.npy"

SHAPE=128

img_test = get_np_arrays(path2)
lab_test = get_np_arrays(path3)
pred_model = get_np_arrays(path4)

# path = r"C:\Users\Mattia\Desktop\Train_images"
# path1 =  r"C:\Users\Mattia\Desktop\Train_labels"

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

image_list=image_list[50:55]
label_list=label_list[50:55]


img_test = img_test[50:55]
lab_test = lab_test[50:55]
pred_model = pred_model[50:55]

pred_model = decode_predictions(pred_model,SHAPE)
pred_model = decode_masks(pred_model,SHAPE)
print('Image and label lists dimensions')
print(len(image_list))
print(len(label_list))

crop_images_list=[]
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')

i = random.randint(0,4) #229
print(i)
image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
image = image.astype('float32')
image/=255                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
resized_image = cv2.resize(image, (SHAPE, SHAPE))
print(resized_image.shape)
print('[INFO]Generating labels array')
    
label = cv2.imread(label_list[i])[:,:,[2,1,0]]
label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#label=np.expand_dims(label, axis=2)  
label = label.astype('float32')
resized_label = cv2.resize(label, (SHAPE, SHAPE), 0, 0, interpolation = cv2.INTER_NEAREST)
resized_label=np.expand_dims(resized_label, axis=2)  
print(resized_label.shape)
img_real=cv2.imread(image_list[i])[:,:,[2,1,0]] 
cv2.imshow('real image',img_real)
cv2.waitKey(0)


soil=0;
bedrock=1;
sand=2;
bigrock=3;
null=255;
label_real=cv2.imread(label_list[i])[:,:,[2,1,0]] 
lab = np.empty((1024, 1024, 3), dtype=np.uint8) 
for r in range(0,1024):
    for c in range(0,1024): 
        channels_xy = label_real[r,c];          #SOIL is kept black, NULL (no label) is white 
        if channels_xy[0]==bedrock:      #BEDROCK --->BLUE
            lab[r,c,0]=255
            lab[r,c,1]=0
            lab[r,c,2]=0
        elif channels_xy[0]==sand:    #SAND --->GREEN
            lab[r,c,0]=0
            lab[r,c,1]=255
            lab[r,c,2]=0
        elif channels_xy[0]==bigrock:    #BIG ROCK ---> RED
            lab[r,c,0]=0
            lab[r,c,1]=0
            lab[r,c,2]=255
        elif channels_xy[0]==soil:    #SOIL ---> BLACK
            lab[r,c,0]=0
            lab[r,c,1]=0
            lab[r,c,2]=0
        elif channels_xy[0]==null:    #NULL ---> WHITE
            lab[r,c,0]=255
            lab[r,c,1]=255
            lab[r,c,2]=255
#lab=cv2.resize(lab,(512,512))

cv2.imshow('real label',lab)
cv2.waitKey(0)


# resized_image = np.asarray(resized_image, np.float32)
# resized_label = np.asarray(resized_label, np.float32)
real = img_real[0:1024,0:1024]
label = lab[0:1024,0:1024]
overlay_img = cv2.addWeighted(real, 1, label,  0.4, 0)
result = cv2.resize(overlay_img, (512, 512))
cv2.imshow('overlay',result)
cv2.waitKey(0)


resized = img_test[i]
print(resized.shape)
predictions = pred_model[i]
print(predictions.shape)
resized = np.asarray(resized, np.float32)
predictions = np.asarray(predictions, np.float32)

overlay_img = cv2.addWeighted(resized, 1, predictions,  0.002, 0)
result = cv2.resize(overlay_img, (512, 512))
cv2.imshow('overlay',result)
cv2.waitKey(0)

null_image = decode_null(resized_label,128)
null_image = np.asarray(null_image, np.float32)
label_new = New_label(null_image,result,128)
#label_new = cv2.resize(label_new, (512, 512))
label_new = np.asarray(label_new, np.float32)
# cv2.imshow('label_new',label_new)
# cv2.waitKey(0)

overlay_final2 = cv2.addWeighted(label_new, 0.002,img_test[i], 1, 0)
overlay_final2 = cv2.resize(overlay_final2, (512, 512))
cv2.imshow('overlay',overlay_final2)
cv2.waitKey(0)

