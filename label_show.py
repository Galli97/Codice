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
# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
# path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_labels_3.npy"

# path = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels_3.npy"

# path = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_labels.npy"
path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print('tmp1: ',tmp1.shape)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
print('tmp2: ',tmp2.shape)

# tmp3 = get_np_arrays(path2)          #recupero tmp1 dal file 
# print('tmp3: ',tmp3.shape)

# tmp4 = get_np_arrays(path3)          #recupero tmp2 dal file
# print('tmp4: ',tmp4.shape)

# tmp5 = get_np_arrays(path4)          #recupero tmp1 dal file 
# print('tmp5: ',tmp5.shape)

# tmp6 = get_np_arrays(path5)          #recupero tmp2 dal file
# print('tmp6: ',tmp6.shape)

# tmp7 = get_np_arrays(path6)          #recupero tmp1 dal file 
# print('tmp5: ',tmp7.shape)

# tmp8 = get_np_arrays(path7)          #recupero tmp2 dal file
# print('tmp6: ',tmp8.shape)


# tmp1=np.concatenate((tmp1,tmp3))#,tmp5))#,tmp7))
# tmp2=np.concatenate((tmp2,tmp4))#,tmp6))#,tmp8))

# print('tmp1_new: ',tmp1.shape)
# print('tmp2_new: ',tmp2.shape)
tmp1=tmp1[100:120]
tmp2=tmp2[100:120]
SHAPE=128;

print(tmp2[10])
true = decode_masks(tmp2,SHAPE)



foto = random.randint(0,len(tmp1)-1)
print('foto: ',foto)

img=cv2.resize(true[foto],(512,512))
img_prediction=cv2.resize(true[10],(512,512))
label=cv2.resize(true[5],(512,512))
# IM=cv2.resize(tmp1[foto],(512,512))

cv2.imshow('image', img) 
cv2.waitKey(0) 
# cv2.imshow('real',IM)
# cv2.waitKey(0) 
cv2.imshow('label',label )
cv2.waitKey(0) 
cv2.imshow('predictions',img_prediction)
cv2.waitKey(0) 

# true_image=tmp1[5]
# true_image = np.asarray(true_image, np.float64)
# print(true_image.shape)
# immagine=cv2.resize(true_image,(512,512))
# print(immagine.shape)
# cv2.imshow('image', immagine) 
# cv2.waitKey(0) 

# label_img=cv2.resize(true[5],(512,512))
# label_img = np.asarray(label_img, np.float64)
# print(label_img.shape)
# cv2.imshow('image', label_img) 
# cv2.waitKey(0) 

# overlay = cv2.addWeighted(immagine, 1, label_img, 0.0008, 0)
# cv2.imshow('overlay',overlay)
# cv2.waitKey(0) 
