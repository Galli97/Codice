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
# path = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_images.npy"
# path1 = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_labels.npy"
# path2 = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_images_2.npy"
# path3 = r"C:\Users\Mattia\Desktop\Dataset_1\Dataset_1\final_labels_2.npy"
# path4= r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
# path5 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
path = r"C:\Users\Mattia\Desktop\Datase_BigRock\Dataset_Bigrock\final_images.npy"
path1 = r"C:\Users\Mattia\Desktop\Datase_BigRock\Dataset_Bigrock\final_labels.npy"
path2= r"C:\Users\Mattia\Documenti\Github\Codice\image_patches.npy"
path3 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches.npy"
### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
print('tmp1: ',tmp1.shape)

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
print('tmp2: ',tmp2.shape)

tmp3 = get_np_arrays(path2)          #recupero tmp1 dal file 
print('tmp3: ',tmp3.shape)

tmp4 = get_np_arrays(path3)          #recupero tmp2 dal file
print('tmp4: ',tmp4.shape)

# tmp5 = get_np_arrays(path4)          #recupero tmp1 dal file 
# print('tmp5: ',tmp5.shape)

# tmp6 = get_np_arrays(path5)          #recupero tmp2 dal file
# print('tmp6: ',tmp6.shape)

# tmp7 = get_np_arrays(path6)          #recupero tmp1 dal file 
# print('tmp5: ',tmp7.shape)

# tmp8 = get_np_arrays(path7)          #recupero tmp2 dal file
# print('tmp6: ',tmp8.shape)


tmp1=np.concatenate((tmp1,tmp3))#,tmp5))#,tmp7))
tmp2=np.concatenate((tmp2,tmp4))#,tmp6))#,tmp8))

print('Image and label lists dimensions')
print('tmp1_new: ',tmp1.shape)
print('tmp2_new: ',tmp2.shape)

####RESHUFFLE DELLA LISTA DELLE IMMAGINI E DELLE LABEL####
image_list, label_list = shuffle(tmp1, tmp2)


save_final_images(image_list)
save_final_labels(label_list)
