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

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)


# fix_gpu()

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

    return K.sum(tf.compat.v1.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.compat.v1.to_float(legal_labels))

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
tmp1 = np.empty((N, 64, 64, 3), dtype=np.uint8)
tmp2 = np.empty((N, 64, 64, 5), dtype=np.uint8)

###### RIEMPIO LE DUE LISTE CON I CORRISPETTIVI ARRAY SFRUTTANDO I PATH SALVATI NELLE PRIME DUE LISTE #######
for i in range (len(image_list)):
    image = cv2.imread(image_list[i])[:,:,[2,1,0]]
    image = cv2.resize(image, (64,64))
    #print(image.shape)
    tmp1[i] = image

bedrock=[1,1,1];
sand=[2,2,2];
bigrock=[3,3,3];
soil=[255,255,255];
nullo=[0,0,0];

### PER LE LABEL CREO UN ARRAY DI DIMENSIONE 64X64X5 (NEW_LABEL) DOVE 64X64 è LA DIMENSIONE DELL'IMMAGINE
### MENTRE 5 è IL NUMERO DI CLASSI. IN QUESTO MODO HO UN VETTORE DEL TIPO [0 0 1 0 0] PER OGNI PIXEL, CHE INDICA
### A QUALE CLASSE APPARTIENE IL PIXEL (IN QUESTO CASO, ALLA TERZA CLASSE). 
for j in range (len(label_list)):
    label = cv2.imread(label_list[j])[:,:,[2,1,0]]
    label = cv2.resize(label, (64,64))
    print(label.shape)
    reduct_label=label[:,:,0]
    print(reduct_label.shape)
    new_label = np.empty((64, 64, 5), dtype=np.uint8)
    for t in range(0,4):
        new_label[:,:,t]=reduct_label
    for i in range(0,63):
        for n in range(0,63): 
            channels_xy = label[i,n];         
            #print(channels_xy)
            if all(channels_xy==bedrock):      #BEDROCK
                new_label[i,n,0]=1
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=0
                #print(new_label.shape)
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
                
            elif all(channels_xy==nullo):    #NULL
                new_label[i,n,0]=0
                new_label[i,n,1]=0
                new_label[i,n,2]=0
                new_label[i,n,3]=0
                new_label[i,n,4]=1
    print(new_label.shape)
    tmp2[j] = new_label
    print(tmp2.shape)
    


##### ALCUNI PRINT DI CONTROLLO ######
#print(len(image_list))
#print(len(label_list))

#print(image_list)
#print(label_list)

#print(len(tmp1))
print(tmp2)

#shape=(64,64,3)
#print(shape)

#model = rete(input_shape=shape,weight_decay=0., classes=5)

#x_train = datagenerator(tmp1,tmp2,2)

#optimizer = SGD(learning_rate=0.01, momentum=0.9)
#loss_fn=softmax_sparse_crossentropy_ignoring_last_label
#loss_fn = keras.losses.CategoricalCrossentropy()
#metrics=[sparse_accuracy_ignoring_last_label]
#metrics=[tf.keras.metrics.MeanIoU(num_classes=5)]

#model.compile(optimizer = optimizer, loss = loss_fn , metrics = ["accuracy"])
#model.compile(loss=loss_fn, optimizer=optimizer,metrics=metrics)
#model.summary()
#model.fit(x = tmp1,y=tmp2,epochs=2,steps_per_epoch=7)