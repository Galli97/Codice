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

print('Image and label lists dimensions')
print(len(image_list))
print(len(label_list))

crop_images_list=[]
###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
i = random.randint(0,len(image_list)-1) #229
print(i)
image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
image = image.astype('float32')
image/=255                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
for r in range (0,8):
    for c in range (0,8):
        cropped_image = image[128*r:128*(r+1),128*c:128*(c+1)]
        crop_images_list.append(cropped_image) 
        
crop_labels_list=[]

print('[INFO]Generating labels array')
    
label = cv2.imread(label_list[i])[:,:,[2,1,0]]
label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
label=np.expand_dims(label, axis=2)  
label = label.astype('float32')
for r in range (0,8):
    for c in range (0,8):
        cropped_label = label[128*(r):128*(r+1),128*(c):128*(c+1)]
        crop_labels_list.append(cropped_label)
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

SHAPE=128;

tmp1 = crop_images_list
tmp2 = decode_labels_overlay(crop_labels_list,SHAPE)

# predictions = decode_predictions(predictions,SHAPE)
# predictions = decode_masks(predictions,SHAPE)

overlay=[];

for j in range (len(tmp2)):
    true_image = tmp1[j]
    true_image = np.asarray(true_image, np.float32)
    label_img = tmp2[j]
    label_img = np.asarray(label_img, np.float32)
    overlay_img = cv2.addWeighted(true_image, 1, label_img,  0.002, 0)
    overlay.append(overlay_img)


img1 = np.hstack((overlay[0], overlay[1],overlay[2], overlay[3],overlay[4], overlay[5],overlay[6], overlay[7]))
img2 = np.hstack((overlay[0+8],overlay[1+8], overlay[2+8],overlay[3+8], overlay[4+8],overlay[5+8], overlay[6+8],overlay[7+8]))
img3 = np.hstack((overlay[0+2*8],overlay[1+2*8], overlay[2+2*8],overlay[3+2*8], overlay[4+2*8],overlay[5+2*8], overlay[6+2*8],overlay[7+2*8]))
img4 = np.hstack((overlay[0+3*8],overlay[1+3*8], overlay[2+3*8],overlay[3+3*8], overlay[4+3*8],overlay[5+3*8], overlay[6+3*8],overlay[7+3*8]))
img5 = np.hstack((overlay[0+4*8],overlay[1+4*8], overlay[2+4*8],overlay[3+4*8], overlay[4+4*8],overlay[5+4*8], overlay[6+4*8],overlay[7+4*8]))
img6 = np.hstack((overlay[0+5*8],overlay[1+5*8], overlay[2+5*8],overlay[3+5*8], overlay[4+5*8],overlay[5+5*8], overlay[6+5*8],overlay[7+5*8]))
img7 = np.hstack((overlay[0+6*8],overlay[1+6*8], overlay[2+6*8],overlay[3+6*8], overlay[4+6*8],overlay[5+6*8], overlay[6+6*8],overlay[7+6*8]))
img8 = np.hstack((overlay[0+7*8],overlay[1+7*8], overlay[2+7*8],overlay[3+7*8], overlay[4+7*8],overlay[5+7*8], overlay[6+7*8],overlay[7+7*8]))

result = np.vstack((img1,img2,img3,img4,img5,img6,img7,img8))

result = cv2.resize(result, (512, 512))
cv2.imshow('overlay',result)
cv2.waitKey(0)

tmp3 = crop_labels_list
BATCH=1
x_test = tf.data.Dataset.from_tensor_slices((tmp1, tmp3))
x_test = (
    x_test
    .batch(BATCH)
)

model = tf.keras.models.load_model('model.h5',custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU,"BilinearInitializer":BilinearInitializer })

print("[INFO] Starting Evaluation")

from memory_profiler import profile      #The output displays the memory consumed by each line in the code. Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. 
# instantiating the decorator
@profile
def prediction(x_test,tmp2):
   predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
   return predictions

predictions = prediction(x_test,tmp2)

predictions = decode_predictions(predictions,SHAPE)
predictions = decode_masks(predictions,SHAPE)


overlay=[];
pred = [];
for d in range (len(tmp2)):
    true_image = tmp1[d]
    true_image = np.asarray(true_image, np.float32)
    label_img = predictions[d]
    label_img = np.asarray(label_img, np.float32)
    overlay_img = cv2.addWeighted(true_image, 1, label_img, 0.002, 0)
    overlay.append(overlay_img)
    pred.append(label_img)

img1 = np.hstack((overlay[0], overlay[1],overlay[2], overlay[3],overlay[4], overlay[5],overlay[6], overlay[7]))
img2 = np.hstack((overlay[0+8],overlay[1+8], overlay[2+8],overlay[3+8], overlay[4+8],overlay[5+8], overlay[6+8],overlay[7+8]))
img3 = np.hstack((overlay[0+2*8],overlay[1+2*8], overlay[2+2*8],overlay[3+2*8], overlay[4+2*8],overlay[5+2*8], overlay[6+2*8],overlay[7+2*8]))
img4 = np.hstack((overlay[0+3*8],overlay[1+3*8], overlay[2+3*8],overlay[3+3*8], overlay[4+3*8],overlay[5+3*8], overlay[6+3*8],overlay[7+3*8]))
img5 = np.hstack((overlay[0+4*8],overlay[1+4*8], overlay[2+4*8],overlay[3+4*8], overlay[4+4*8],overlay[5+4*8], overlay[6+4*8],overlay[7+4*8]))
img6 = np.hstack((overlay[0+5*8],overlay[1+5*8], overlay[2+5*8],overlay[3+5*8], overlay[4+5*8],overlay[5+5*8], overlay[6+5*8],overlay[7+5*8]))
img7 = np.hstack((overlay[0+6*8],overlay[1+6*8], overlay[2+6*8],overlay[3+6*8], overlay[4+6*8],overlay[5+6*8], overlay[6+6*8],overlay[7+6*8]))
img8 = np.hstack((overlay[0+7*8],overlay[1+7*8], overlay[2+7*8],overlay[3+7*8], overlay[4+7*8],overlay[5+7*8], overlay[6+7*8],overlay[7+7*8]))

result = np.vstack((img1,img2,img3,img4,img5,img6,img7,img8))

pred1 = np.hstack((pred[0], pred[1],pred[2], pred[3],pred[4], pred[5],pred[6], pred[7]))
pred2 = np.hstack((pred[0+8],pred[1+8], pred[2+8],pred[3+8], pred[4+8],pred[5+8], pred[6+8],pred[7+8]))
pred3 = np.hstack((pred[0+2*8],pred[1+2*8], pred[2+2*8],pred[3+2*8], pred[4+2*8],pred[5+2*8], pred[6+2*8],pred[7+2*8]))
pred4 = np.hstack((pred[0+3*8],pred[1+3*8], pred[2+3*8],pred[3+3*8], pred[4+3*8],pred[5+3*8], pred[6+3*8],pred[7+3*8]))
pred5 = np.hstack((pred[0+4*8],pred[1+4*8], pred[2+4*8],pred[3+4*8], pred[4+4*8],pred[5+4*8], pred[6+4*8],pred[7+4*8]))
pred6 = np.hstack((pred[0+5*8],pred[1+5*8], pred[2+5*8],pred[3+5*8], pred[4+5*8],pred[5+5*8], pred[6+5*8],pred[7+5*8]))
pred7 = np.hstack((pred[0+6*8],pred[1+6*8], pred[2+6*8],pred[3+6*8], pred[4+6*8],pred[5+6*8], pred[6+6*8],pred[7+6*8]))
pred8 = np.hstack((pred[0+7*8],pred[1+7*8], pred[2+7*8],pred[3+7*8], pred[4+7*8],pred[5+7*8], pred[6+7*8],pred[7+7*8]))

result_pred = np.vstack((pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8))

result = cv2.resize(result, (512, 512))
cv2.imshow('overlay_prediction',result)
cv2.waitKey(0)

null_image = decode_null(label_real,1024)
null_image = np.asarray(null_image, np.float32)
# cv2.imshow('overlay',null_image)
# cv2.waitKey(0)

# result_pred = np.asarray(result_pred, np.float32)
# overlay_final = cv2.addWeighted(result_pred, 1, null_image, 0.005, 0)
# cv2.imshow('overlay2',overlay_final)
# cv2.waitKey(0)
#overlay_final = np.asarray(overlay_final, np.float32)

label_new = New_label(null_image,result_pred,1024)

img_real=cv2.imread(image_list[i])[:,:,[2,1,0]]
# cv2.imshow('label_new',label_new)
# cv2.waitKey(0)

overlay2=[]
for d in range (len(tmp2)):
    true_image = tmp1[d]
    true_image = np.asarray(true_image, np.float32)
    overlay2.append(true_image)

img_1 = np.hstack((overlay2[0], overlay2[1],overlay2[2], overlay2[3],overlay2[4], overlay2[5],overlay2[6], overlay2[7]))
img_2 = np.hstack((overlay2[0+8],overlay2[1+8], overlay2[2+8],overlay2[3+8], overlay2[4+8],overlay2[5+8], overlay2[6+8],overlay2[7+8]))
img_3 = np.hstack((overlay2[0+2*8],overlay2[1+2*8], overlay2[2+2*8],overlay2[3+2*8], overlay2[4+2*8],overlay2[5+2*8], overlay2[6+2*8],overlay2[7+2*8]))
img_4 = np.hstack((overlay2[0+3*8],overlay2[1+3*8], overlay2[2+3*8],overlay2[3+3*8], overlay2[4+3*8],overlay2[5+3*8], overlay2[6+3*8],overlay2[7+3*8]))
img_5 = np.hstack((overlay2[0+4*8],overlay2[1+4*8], overlay2[2+4*8],overlay2[3+4*8], overlay2[4+4*8],overlay2[5+4*8], overlay2[6+4*8],overlay2[7+4*8]))
img_6 = np.hstack((overlay2[0+5*8],overlay2[1+5*8], overlay2[2+5*8],overlay2[3+5*8], overlay2[4+5*8],overlay2[5+5*8], overlay2[6+5*8],overlay2[7+5*8]))
img_7 = np.hstack((overlay2[0+6*8],overlay2[1+6*8], overlay2[2+6*8],overlay2[3+6*8], overlay2[4+6*8],overlay2[5+6*8], overlay2[6+6*8],overlay2[7+6*8]))
img_8 = np.hstack((overlay2[0+7*8],overlay2[1+7*8], overlay2[2+7*8],overlay2[3+7*8], overlay2[4+7*8],overlay2[5+7*8], overlay2[6+7*8],overlay2[7+7*8]))

result2 = np.vstack((img_1,img_2,img_3,img_4,img_5,img_6,img_7,img_8))

label_new = np.asarray(label_new, np.float32)
overlay_final2 = cv2.addWeighted(label_new, 0.002,result2, 1, 0)
overlay_final2 = cv2.resize(overlay_final2, (512, 512))
cv2.imshow('overlay3',overlay_final2)
cv2.waitKey(0)