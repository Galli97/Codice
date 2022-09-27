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
# path = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\Dataset_1\510Data\cropped_labels_3.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi128_2\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128_2\label_patches_TEST.npy"

# path = r"C:\Users\Mattia\Desktop\Datase_BigRock\Dataset_Bigrock\final_images.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Datase_BigRock\Dataset_Bigrock\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\Dataset_Resize\Selezionate2000\Dataset_resize\final_images.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Dataset_Resize\Selezionate2000\Dataset_resize\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\Resized_Test\Test510\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Resized_Test\Test510\label_patches_TEST.npy"

# path = r"C:\Users\Mattia\Desktop\Resize512_crop128\Dataset_Res512\final_images.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Resize512_crop128\Dataset_Res512\final_labels.npy"

path = r"C:\Users\Mattia\Desktop\Resize256_crop128\Dataset_Res256\final_images.npy"
path1 =  r"C:\Users\Mattia\Desktop\Resize256_crop128\Dataset_Res256\final_labels.npy"

# path = r"C:\Users\Mattia\Desktop\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\label_patches_TEST.npy"


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

# # tmp7 = get_np_arrays(path6)          #recupero tmp1 dal file 
# # print('tmp5: ',tmp7.shape)

# # tmp8 = get_np_arrays(path7)          #recupero tmp2 dal file
# # print('tmp6: ',tmp8.shape)


# tmp1=np.concatenate((tmp1,tmp3,tmp5))#,tmp7))
# tmp2=np.concatenate((tmp2,tmp4,tmp6))#,tmp8))

# print('Image and label lists dimensions')
# print('tmp1_new: ',tmp1.shape)
# print('tmp2_new: ',tmp2.shape)

# tmp1=tmp1[0:1500]
# tmp2=tmp2[0:1500]

# ####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
# image_list, label_list = shuffle(tmp1, tmp2)
print(tmp1[0,0,0,0])
for j in range (0,len(tmp1)):
    image = tmp1[j]
    image*=2
    tmp1[j]=image

print(tmp1[0,0,0,0])
######## SALVATAGGIO ####
# print("[INFO] Cropped images arrays saved")
# save_cropped_images(tmp1) 

# ######## SALVATAGGIO ####
# print("[INFO] Cropped labels arrays saved")
# save_cropped_labels(tmp2) 


# ######## SALVATAGGIO ####
# print("[INFO] Cropped images arrays saved")
# save_patches_TEST(tmp1) 

# ######## SALVATAGGIO ####
# print("[INFO] Cropped labels arrays saved")
# save_label_patches_TEST(tmp2) 

# ######## SALVATAGGIO ####
# print("[INFO] Cropped images arrays saved")
# save_final_images(tmp1) 

# ######## SALVATAGGIO ####
# print("[INFO] Cropped labels arrays saved")
# save_final_labels(tmp2)

save_final_images(tmp1)
save_final_labels(tmp2)