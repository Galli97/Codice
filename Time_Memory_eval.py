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



path = r"C:\Users\Mattia\Desktop\TEST_images"
path1 =  r"C:\Users\Mattia\Desktop\TEST_labels"

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

SHAPE=128;
model = tf.keras.models.load_model('model.h5',custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU })

###### RIEMPIO LA LISTA IMMAGINI CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI IN IMAGE_LIST #######
print('[INFO]Generating images array')
print('[INFO]Generating labels array')
import time
tempo=0;
steps=3;
for i in range (0,steps):
    start_time = time.time()
    crop_images_list=[]
    crop_labels_list=[]
    print(i)
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
    image = image.astype('float32')
    image/=255                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
    for r in range (0,8):
        for c in range (0,8):
            cropped_image = image[128*r:128*(r+1),128*c:128*(c+1)]
            crop_images_list.append(cropped_image) 
    label = cv2.imread(label_list[i])[:,:,[2,1,0]]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label=np.expand_dims(label, axis=2)  
    label = label.astype('float32')
    for r in range (0,8):
        for c in range (0,8):
            cropped_label = label[128*(r):128*(r+1),128*(c):128*(c+1)]
            crop_labels_list.append(cropped_label)

    tmp1 = crop_images_list
    tmp2 = crop_labels_list

    BATCH=1
    x_test = tf.data.Dataset.from_tensor_slices((tmp1, tmp2))
    x_test = (
        x_test
        .batch(BATCH)
    )

    print("[INFO] Starting Evaluation")

    from memory_profiler import profile      #The output displays the memory consumed by each line in the code. Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. 
    from memory_profiler import memory_usage 
    # instantiating the decorator            #1 MiB (mebibyte) is always 1024 Kb
    @profile
    def prediction(x_test,tmp2):
        predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
        return predictions

    predictions = prediction(x_test,tmp2)
    tempo += time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))
    # import os, psutil
    # process = psutil.Process(os.getpid())
    # #print(process.memory_info().vms)
    # print('MB: ',psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2)
    # print('The CPU usage is: ', psutil.cpu_percent(4))
    # print('RAM memory % used:', psutil.virtual_memory().percent)
    

tempo_medio = tempo/steps;

print('Model with cropped images: ',tempo_medio)


##############RESIZE############
# tmp1 = np.empty((1, SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
# tmp2 = np.empty((1, SHAPE, SHAPE, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe
# tempo=0;
# steps=3;
# for i in range (0,steps):
#     start_time = time.time()
#     crop_images_list=[]
#     crop_labels_list=[]
#     print(i)
#     image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
#     image = image.astype('float32')
#     image/=255                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
#     resized_image = cv2.resize(image, (SHAPE, SHAPE))
#     label = cv2.imread(label_list[i])[:,:,[2,1,0]]
#     label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
#     label = label.astype('float32')
#     resized_label = cv2.resize(label, (SHAPE, SHAPE), 0, 0, interpolation = cv2.INTER_NEAREST)
#     resized_label=np.expand_dims(resized_label, axis=2) 
#     tmp1[0] = resized_image
#     tmp2[0] = resized_label

#     BATCH=1
#     x_test = tf.data.Dataset.from_tensor_slices((tmp1, tmp2))
#     x_test = (
#         x_test
#         .batch(BATCH)
#     )

#     print("[INFO] Starting Evaluation")

#     from memory_profiler import profile      #The output displays the memory consumed by each line in the code. Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. 
#     # instantiating the decorator            #1 MiB (mebibyte) is always 1024 Kb
#     @profile
#     def prediction(x_test,tmp2):
#         predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
#         return predictions

#     predictions = prediction(x_test,tmp2)
#     tempo += time.time() - start_time
#     print("--- %s seconds ---" % (time.time() - start_time))
# tempo_medio = tempo/steps;

# print('Model with resized images: ',tempo_medio)

###############FUNZIONE INTERA ###########
# from memory_profiler import profile      #The output displays the memory consumed by each line in the code. Implementation of finding the memory consumption is very easy using a memory profiler as we directly call the decorator instead of writing a whole new code. 
# # instantiating the decorator            #1 MiB (mebibyte) is always 1024 Kb
# @profile
# def process(steps,image_list,label_list,SHAPE):
#         import time
#         tempo=0;
#         for i in range (0,steps):
        
#             tmp1 = np.empty((1, SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
#             tmp2 = np.empty((1, SHAPE, SHAPE, 1), dtype=np.uint8)  #Qui ho N labels, che portano l'informazione per ogni pixel. Nel caso sparse avrò un intero ad indicare la classe

#             start_time = time.time()
#             crop_images_list=[]
#             crop_labels_list=[]
#             print(i)
#             image = cv2.imread(image_list[i])[:,:,[2,1,0]]  #leggo le immagini
#             image = image.astype('float32')
#             image/=255                                    #normalizzo per avere valori per i pixel nell'intervallo [0,0.5]
#             resized_image = cv2.resize(image, (SHAPE, SHAPE))
#             label = cv2.imread(label_list[i])[:,:,[2,1,0]]
#             label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
#             label = label.astype('float32')
#             resized_label = cv2.resize(label, (SHAPE, SHAPE), 0, 0, interpolation = cv2.INTER_NEAREST)
#             resized_label=np.expand_dims(resized_label, axis=2) 
#             tmp1[0] = resized_image
#             tmp2[0] = resized_label

#             BATCH=1
#             x_test = tf.data.Dataset.from_tensor_slices((tmp1, tmp2))
#             x_test = (
#                 x_test
#                 .batch(BATCH)
#             )

#             print("[INFO] Starting Evaluation")

        
#             #def prediction(x_test,tmp2):
#             predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
#             #    return predictions

#             #predictions = prediction(x_test,tmp2)
#             tempo += time.time() - start_time
#             print("--- %s seconds ---" % (time.time() - start_time))

# process(100,image_list,label_list,SHAPE)

    

