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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.preprocessing.image import ImageDataGenerator
####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\Tentativi128\DATASET\final_labels_3.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_1\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_1\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_2\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_2\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_3\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_3\final_labels_3.npy"
# path6 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_4\final_images_4.npy"
# path7 = r"C:\Users\Mattia\Desktop\Tentativi\Trial24\DATASET_4\final_labels_4.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Tentativi128_2\DATASET\final_labels_2.npy"

# path = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\TentativiBR_128\DATASET\final_labels_3.npy"

# path = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels_2.npy"
# path4 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images_3.npy"
# path5 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels_3.npy"
# path6 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images_4.npy"
# path7 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels_4.npy"
# path8 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images_5.npy"
# path9 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels_5.npy"
# path10 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_images_6.npy"
# path11 = r"C:\Users\Mattia\Desktop\DatasetNew\DatasetNew\final_labels_6.npy"

path = r"C:\Users\Mattia\Desktop\DatasetSoil\final_images.npy"
path1 = r"C:\Users\Mattia\Desktop\DatasetSoil\final_labels.npy"
path2 = r"C:\Users\Mattia\Desktop\DatasetSoil\final_images_2.npy"
path3 = r"C:\Users\Mattia\Desktop\DatasetSoil\final_labels_2.npy"
path4 = r"C:\Users\Mattia\Desktop\DatasetSoil\final_images_3.npy"
path5 = r"C:\Users\Mattia\Desktop\DatasetSoil\final_labels_3.npy"
# path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"
# path = r"C:\Users\Mattia\Desktop\Tentativi128_2\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128_2\label_patches_TEST.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print('tmp1: ',tmp1.shape)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
print('tmp2: ',tmp2.shape)

tmp3 = get_np_arrays(path2)          #recupero tmp1 dal file 
print('tmp3: ',tmp3.shape)

tmp4 = get_np_arrays(path3)          #recupero tmp2 dal file
print('tmp4: ',tmp4.shape)

tmp5 = get_np_arrays(path4)          #recupero tmp1 dal file 
print('tmp5: ',tmp5.shape)

tmp6 = get_np_arrays(path5)          #recupero tmp2 dal file
print('tmp6: ',tmp6.shape)

# tmp7 = get_np_arrays(path6)          #recupero tmp1 dal file 
# print('tmp7: ',tmp7.shape)

# tmp8 = get_np_arrays(path7)          #recupero tmp2 dal file
# print('tmp8: ',tmp8.shape)

# tmp9 = get_np_arrays(path8)          #recupero tmp1 dal file 
# print('tmp9: ',tmp9.shape)

# tmp10 = get_np_arrays(path9)          #recupero tmp2 dal file
# print('tmp10: ',tmp10.shape)

# tmp11 = get_np_arrays(path10)          #recupero tmp1 dal file 
# print('tmp11: ',tmp11.shape)

# tmp12 = get_np_arrays(path11)          #recupero tmp2 dal file
# print('tmp12: ',tmp12.shape)


tmp1=np.concatenate((tmp1,tmp3,tmp5))#,tmp7,tmp9,tmp11))
tmp2=np.concatenate((tmp2,tmp4,tmp6))#,tmp8,tmp10,tmp12))

print('tmp1_new: ',tmp1.shape)
print('tmp2_new: ',tmp2.shape)

train_set = int(len(tmp2)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]
print('list_train: ',list_train.shape)
print('list_validation: ',list_validation.shape)

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]
print('label_train: ',label_train.shape)
print('label_validation: ',label_validation.shape)

soil_count=0;
bedrock_count=0;
sand_count=0;
bigrock_count=0;
null_count=0;

SHAPE=128;
# for i in range (0,len(tmp2)):
#     for r in range(0,SHAPE):
#         for c in range (0,SHAPE):
#             # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
#             #     print(tmp1[i,r,c,:])
#             if tmp2[i,r,c,:]==4:
#                 soil_count+=1
#             elif tmp2[i,r,c,:]==1:
#                 bedrock_count+=1
#             elif tmp2[i,r,c,:]==2:
#                 sand_count+=1
#             elif tmp2[i,r,c,:]==3:
#                 bigrock_count+=1
#             elif tmp2[i,r,c,:]==0:
#                 null_count+=1
#             else:
#                 print(i)


for i in range (0,len(label_train)):
    for r in range(0,SHAPE):
        for c in range (0,SHAPE):
            # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
            #     print(tmp1[i,r,c,:])
            if label_train[i,r,c,:]==4:
                soil_count+=1
            if label_train[i,r,c,:]==1:
                bedrock_count+=1
            if label_train[i,r,c,:]==2:
                sand_count+=1
            if label_train[i,r,c,:]==3:
                bigrock_count+=1
            if label_train[i,r,c,:]==0:
                null_count+=1

print('null: ', null_count)
print('bedrock: ', bedrock_count)
print('sand: ', sand_count)
print('bigrock: ', bigrock_count)
print('soil: ', soil_count)



soil_count_val=0;
bedrock_count_val=0;
sand_count_val=0;
bigrock_count_val=0;
null_count_val=0;

for i in range (0,len(label_validation)):
    for r in range(0,SHAPE):
        for c in range (0,SHAPE):
            # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
            #     print(tmp1[i,r,c,:])
            if label_validation[i,r,c,:]==4:
                soil_count_val+=1
            if label_validation[i,r,c,:]==1:
                bedrock_count_val+=1
            if label_validation[i,r,c,:]==2:
                sand_count_val+=1
            if label_validation[i,r,c,:]==3:
                bigrock_count_val+=1
            if label_validation[i,r,c,:]==0:
                null_count_val+=1

print('null_val: ', null_count_val)
print('bedrock_val: ', bedrock_count_val)
print('sand_val: ', sand_count_val)
print('bigrock_val: ', bigrock_count_val)
print('soil_val: ', soil_count_val)