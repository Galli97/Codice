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
import matplotlib.pyplot as plt
from PIL import Image
from rete import *
from tensorflow.keras.optimizers import SGD
from utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.preprocessing.image import ImageDataGenerator

###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
# path = r"/content/drive/MyDrive/Tesi/image_patches_TEST.npy"
# path1 = r"/content/drive/MyDrive/Tesi/label_patches_TEST.npy"

# ####### PERCORSO IN LOCALE #########
path = r"C:\Users\Mattia\Documenti\Github\Codice\image_patches_TEST.npy"
path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\label_patches_TEST.npy"
path2= r"C:\Users\Mattia\Documenti\Github\Codice\cropped_images_TEST.npy"
path3 =  r"C:\Users\Mattia\Documenti\Github\Codice\cropped_labels_TEST.npy"
### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))
print(tmp1.shape)
print('0: ',tmp1[1])
# tmp1=tmp1[0:50]

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)
print(len(tmp2))
print('0: ',tmp2[8,:,:,0])
# tmp2=tmp2[0:50]
print('tmp2: ', tmp2.shape)
# crop_images_list = get_np_arrays(path2)          #recupero tmp1 dal file 
# crop_labels_list = get_np_arrays(path3) 

shape=(64,64,1)
print(shape)
BATCH= 1
EPOCHS=10

x_test = datagenerator(tmp1,tmp2,BATCH)

model = tf.keras.models.load_model('model.h5',custom_objects={"sparse_accuracy_ignoring_last_label": sparse_accuracy_ignoring_last_label })

print("[INFO] Starting Evaluation")

predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
print(predictions[0])
#print(np.argmax(predictions[0,0,0,:]))
# predictions = np.squeeze(predictions)
# predictions = np.argmax(predictions, axis=2)
print(predictions.shape)
save_predictions(predictions)

#true = decode_masks(tmp2)
prediction = decode_predictions(predictions)
cm1=np.ravel(tmp2)
print(cm1.shape)
cm2=np.ravel(prediction)
print(cm2.shape)
print(prediction[0].shape)
print(prediction[0])
matrix = tf.math.confusion_matrix(cm1,cm2,num_classes=5)
print(matrix)