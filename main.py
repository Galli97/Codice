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

def datagenerator(images,labels, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):

            x = images[start:end] 
            y = labels[start:end]
            yield x,y

            start += batchsize
            end += batchsize

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.compat.v1.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.compat.v1.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.compact.v1.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.compact.v1.to_float(legal_labels))

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If it doesn't work, uncomment this line; it should help.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

####### PERCORSO IN LOCALE #########
#path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
#path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"

####### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
path = r"/content/drive/MyDrive/Tesi/Dataset/Train_images"
path1 = r"/content/drive/MyDrive/Tesi/Dataset/Train_labels"

####### CREO UNA LISTA CON ELEMENTI DATI DA QUELLI NELLA CARTELLA DEL PERCORSO ######
dir = os.listdir(path)
dir1 = os.listdir(path1)

###### INIZIALIZO DUE LISTE ########
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


##### INIZIALIZO DUE LISTE CHE ANDRANNO A CONTENERE GLI ARRAY DELLE IMMAGINI ######
N = len(image_list)
print(N)
tmp1 = np.empty((N, 1024, 1024, 3), dtype=np.uint8)
tmp2 = np.empty((N, 1024,1024, 3), dtype=np.uint8)

###### RIEMPIO LE DUE LISTE CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI NELLE PRIME DUE LISTE #######
for i in range (len(image_list)):
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]
    #print(image.shape)
    tmp1[i] = image

for j in range (len(label_list)):
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    tmp2[j] = label


##### ALCUNI PRINT DI CONTROLLO ######
#print(len(image_list))
#print(len(label_list))

#print(image_list)
#print(label_list)

print(len(tmp1))
print(len(tmp2))

shape=(1024,1024,3)
print(shape)

model = rete(input_shape=shape,weight_decay=0., classes=5)

x_train = datagenerator(tmp1,tmp2,64)

optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn=softmax_sparse_crossentropy_ignoring_last_label
#loss_fn = keras.losses.SparseCategoricalCrossentropy()
metrics=[sparse_accuracy_ignoring_last_label]
#metrics=[tf.keras.metrics.MeanIoU(num_classes=5)]

model.compile(loss=loss_fn, optimizer=optimizer,metrics=metrics)
model.fit(x = x_train,epochs=2,steps_per_epoch=5)