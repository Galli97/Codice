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

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# print(os.getenv('TF_GPU_ALLOCATOR'))
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = InteractiveSession(config=config)
###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
path = r"/content/drive/MyDrive/Tesi/final_images.npy"
path1 = r"/content/drive/MyDrive/Tesi/final_labels.npy"

# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))
print(tmp1.shape)
# print('999: ',tmp1[999])
# print('0: ',tmp1[0])
# print('4000: ',tmp1[4000])
tmp1=tmp1[:700]
# print(tmp1.shape)


tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)
# print(len(tmp2))
# print('999: ',tmp2[999])
# print('0: ',tmp2[0])
# print('3050: ',tmp2[3050])
tmp2=tmp2[:700]
# print(tmp2.shape)


# N = len(tmp2)
# tmp1=tmp1[:N]
# print(tmp1.shape)

# #### PRENDO UNA PARTE DEL DATASET (20%) E LO UTILIZZO PER IL VALIDATION SET #####
train_set = int(len(tmp2)*80/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]
print('list_train: ',list_train.shape)
print('list_validation: ',list_validation.shape)

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]
print('label_train: ',label_train.shape)
print('label_validation: ',label_validation.shape)

# soil:  1895582
# bedrock:  3880619
# sand:  1108513
# bigrock:  1413195
# null:  7037515

null_pixels = 7037515;
bedrock_pixels = 3880619;
sand_pixels = 1108513;
bigrock_pixels = 1413195;
soil_pixels = 1895582;

PIXELS=soil_pixels+bedrock_pixels + sand_pixels+bigrock_pixels+null_pixels ;
loss_weights=[soil_pixels/PIXELS,bedrock_pixels/PIXELS,sand_pixels/PIXELS,bigrock_pixels/PIXELS,null_pixels/PIXELS]
# label_train = label_train.reshape((len(label_train),64*64,1))
# label_validation = label_validation.reshape((len(label_validation),64*64,1))
# print('label_train: ',label_train.shape)
# print('label_validation: ',label_validation.shape)

###### DEFINISCO IL MODELLO #######
shape=(64,64,1)
print(shape)
BATCH=16
EPOCHS = 10
steps = int(train_set/EPOCHS)
weight_decay = 0.0001/2
batch_shape=(BATCH,64,64,1)
model = rete(input_shape=shape,weight_decay=weight_decay,batch_shape=None, classes=5)

#model = DeeplabV3Plus(image_size=64,num_classes=5)

##### USO DATAGENERATOR PER PREPARARE I DATI DA MANDARE NELLA RETE #######
# x_train = datagenerator(list_train,label_train,BATCH)
# x_validation = datagenerator(list_validation,label_validation,BATCH)
#print(type(x_train))

BUFFER_SIZE=train_set;
# # Create a Dataset that includes sample weights
# # (3rd element in the return tuple).
x_train = tf.data.Dataset.from_tensor_slices((list_train, label_train))
x_train = x_train.cache()
x_train = x_train.shuffle(BUFFER_SIZE)
x_train = x_train.batch(BATCH)
x_train = x_train.repeat()
x_train = x_train.prefetch(buffer_size=tf.data.AUTOTUNE)
#print(x_train.shape)
x_validation = tf.data.Dataset.from_tensor_slices((list_validation, label_validation))
x_validation = x_validation.batch(BATCH)
     

x_train = x_train.map(add_sample_weights)
#x_validation = x_validation.map(add_sample_weights)

#### DEFINSICO I PARAMETRI PER IL COMPILE (OPTIMIZER E LOSS)

lr_base = 0.01 * (float(BATCH) / 16)
optimizer = SGD(learning_rate=lr_base, momentum=0.)
#optimizer=keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()#keras.losses.SparseCategoricalCrossentropy(from_logits=True) #iou_coef #softmax_sparse_crossentropy_ignoring_last_label

model.compile(optimizer = optimizer, loss = loss_fn , metrics =['accuracy'],sample_weight_mode='temporal')#tf.keras.metrics.SparseCategoricalAccuracy()s#loss_weights=loss_weights#[tf.keras.metrics.SparseCategoricalAccuracy()]#[tf.keras.metrics.MeanIoU(num_classes=5)])#['accuracy'])#[sparse_accuracy_ignoring_last_label])#,sample_weight_mode='temporal')

### AVVIO IL TRAINING #####
model.summary()
# history = 
model.fit(x = x_train,batch_size=BATCH, steps_per_epoch=steps,epochs=EPOCHS,validation_data=x_validation)
model.save('model.h5')

# plt.plot(history.history["loss"])
# plt.title("Training Loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history['sparse_accuracy_ignoring_last_label'])
# plt.title("Training Accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history["val_loss"])
# plt.title("Validation Loss")
# plt.ylabel("val_loss")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history["val_sparse_accuracy_ignoring_last_label"])
# plt.title("Validation Accuracy")
# plt.ylabel("val_accuracy")
# plt.xlabel("epoch")
# plt.show()

# x_train=tf.keras.applications.vgg16.preprocess_input(x_train)
# x_validation=tf.keras.applications.vgg16.preprocess_input(x_validation)
# sample_weight = np.ones(shape=(len(label_train),64,64))
# print(sample_weight.shape)
# sample_weight[:,0] = 1.5
# sample_weight[:,1] = 0.5
# sample_weight[:,2] = 1.5
# sample_weight[:,3] = 3.0
# sample_weight[:,4] = 0

# val_sample_weight = np.ones(shape=(len(label_validation),64,64))
# print(val_sample_weight.shape)
# val_sample_weight[:,0] = 1.5
# val_sample_weight[:,1] = 0.5
# val_sample_weight[:,2] = 1.5
# val_sample_weight[:,3] = 3.0
# val_sample_weight[:,4] = 0