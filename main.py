import tensorflow as tf
from keras.layers import *
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

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If it doesn't work, uncomment this line; it should help.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
#path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"

path = r"/content/drive/MyDrive/Tesi/Dataset/Train_images"
path1 = r"/content/drive/MyDrive/Tesi/Dataset/Train_labels"

dir = os.listdir(path)
dir1 = os.listdir(path1)

image_list = []
label_list = []

for elem in dir:
    new_dir = os.path.join(path,elem)
    if new_dir not in image_list : image_list.append(new_dir)
    # read the image data using PIL
    #image = Image.open(new_dir)
    #image = np.array(image)
    #image=np.expand_dims(image, axis=2)
    #shape=image.shape
    #print(shape)
    

for lab in dir1:
    new_dir1 = os.path.join(path1,lab)
    if new_dir1 not in label_list : label_list.append(new_dir1)
    # read the image data using PIL
    #label = Image.open(new_dir1)
    #label = np.array(label)
    #label=np.expand_dims(label, axis=2)
    #label_shape=label.shape
    #print(label_shape)
    
for i in image_list:
    image = cv2.imread(image_list[i])
    tmp1[i] = image

for j in label_list:
    label = cv2.imread(label_list[j])
    tmp2[j] = label



#print(len(image_list))
#print(len(label_list))

print(image_list)
print(label_list)

print(tmp1)
print(tmp2)

#model = rete(input_shape=shape,weight_decay=0., classes=5)

#optimizer = SGD(learning_rate=0.01, momentum=0.9)
#loss_fn=softmax_sparse_crossentropy_ignoring_last_label
#metrics=[sparse_accuracy_ignoring_last_label]

#model.compile(loss=loss_fn, optimizer=optimizer,metrics=metrics)
#model.fit(x = list_train,y=labels_train,epochs=2,steps_per_epoch=5)