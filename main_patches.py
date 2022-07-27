import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import image
from keras.callbacks import *
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

config = ConfigProto()
config.gpu_options.allow_growth=True
session = InteractiveSession(config=config)
###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
#64x64
# path = r"/content/drive/MyDrive/Tesi/Dataset64/final_images.npy"
# path1 = r"/content/drive/MyDrive/Tesi/Dataset64/final_labels.npy"
# path2 = r"/content/drive/MyDrive/Tesi/Dataset64/final_images_2.npy"
# path3 = r"/content/drive/MyDrive/Tesi/Dataset64/final_labels_2.npy"
# path4 = r"/content/drive/MyDrive/Tesi/Dataset64/final_images_3.npy"
# path5 = r"/content/drive/MyDrive/Tesi/Dataset64/final_labels_3.npy"
# path6 = r"/content/drive/MyDrive/Tesi/Dataset64/final_images_4.npy"
# path7 = r"/content/drive/MyDrive/Tesi/Dataset64/final_labels_4.npy"

#128x128
path = r"/content/drive/MyDrive/Tesi/Dataset128/final_images.npy"
path1 = r"/content/drive/MyDrive/Tesi/Dataset128/final_labels.npy"
path2 = r"/content/drive/MyDrive/Tesi/Dataset128/final_images_2.npy"
path3 = r"/content/drive/MyDrive/Tesi/Dataset128/final_labels_2.npy"
path4 = r"/content/drive/MyDrive/Tesi/Dataset128/final_images_3.npy"
path5 = r"/content/drive/MyDrive/Tesi/Dataset128/final_labels_3.npy"


# path = r"/content/drive/MyDrive/Tesi/image_patches.npy"
# path1 = r"/content/drive/MyDrive/Tesi/label_patches.npy"

# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Documenti\Github\Codice\final_images.npy"
# path1 =  r"C:\Users\Mattia\Documenti\Github\Codice\final_labels.npy"

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
# print('tmp5: ',tmp7.shape)

# tmp8 = get_np_arrays(path7)          #recupero tmp2 dal file
# print('tmp6: ',tmp8.shape)


tmp1=np.concatenate((tmp1,tmp3,tmp5))#,tmp7))
tmp2=np.concatenate((tmp2,tmp4,tmp6))#,tmp8))

print('tmp1_new: ',tmp1.shape)
print('tmp2_new: ',tmp2.shape)


# print(tmp1.shape)
# print(tmp2.shape)
# #### PRENDO UNA PARTE DEL DATASET (20%) E LO UTILIZZO PER IL VALIDATION SET #####
train_set = int(len(tmp2)*70/100)

list_train = tmp1[:train_set]
list_validation = tmp1[train_set:]
print('list_train: ',list_train.shape)
print('list_validation: ',list_validation.shape)

label_train = tmp2[:train_set]
label_validation = tmp2[train_set:]
print('label_train: ',label_train.shape)
print('label_validation: ',label_validation.shape)



checkpoint_path = "./cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


###### DEFINISCO IL MODELLO #######
SHAPE=128;
shape=(SHAPE,SHAPE,3)
BATCH = 64
EPOCHS = 25
steps = int(train_set/(EPOCHS))
steps_val = int(len(list_validation)/EPOCHS)
weight_decay = 0.0005
batch_shape=(BATCH,SHAPE,SHAPE,3)
#model = rete(input_shape=shape,weight_decay=weight_decay,batch_shape=None, classes=5)
tf.keras.backend.set_image_data_format('channels_last')
model = rete(input_shape=shape,weight_decay=weight_decay,batch_shape=None, classes=5)
#model = rete_vgg16_dilation(img_size=shape,weight_decay=weight_decay,batch_shape=None, classes=5)
input_shape = (SHAPE, SHAPE, 3)
#model = build_vgg16_unet(input_shape,weight_decay=weight_decay, classes=5)

##### USO DATAGENERATOR PER PREPARARE I DATI DA MANDARE NELLA RETE #######
# x_train = datagenerator(list_train,label_train,BATCH)
# x_validation = datagenerator(list_validation,label_validation,BATCH)


BUFFER_SIZE=train_set;
x_train = tf.data.Dataset.from_tensor_slices((list_train, label_train))

x_validation = tf.data.Dataset.from_tensor_slices((list_validation, label_validation))

 
x_train = x_train.map(add_sample_weights)
x_train = (
    x_train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH)
    .repeat(EPOCHS)                         ###Ad ogni epoch avrò un numero di batch pari ha len(dataset)/Batch_size. 
    .prefetch(buffer_size=tf.data.AUTOTUNE))


print(x_train)
x_validation = x_validation.map(add_sample_weights_val)   
x_validation = x_validation.batch(BATCH)

lr_base = 0.01 * (float(BATCH) / 16)

def lr_scheduler(epoch):
  
    # drops as progression proceeds, good for sgd
    if epoch > 0.75 * EPOCHS:
        lr = 0.001
    elif epoch > 0.5 * EPOCHS:
        lr = 0.01
    else:
        lr = 0.01 * (float(BATCH) / 16)
    #print('lr: %f' % lr)
    return lr

scheduler = LearningRateScheduler(lr_scheduler)
callbacks = [scheduler]



optimizer = SGD(learning_rate=lr_base, momentum=0.9)
#optimizer=keras.optimizers.Adam(learning_rate=0.001)
loss_fn =keras.losses.SparseCategoricalCrossentropy()#keras.losses.SparseCategoricalCrossentropy(from_logits=True) #iou_coef #softmax_sparse_crossentropy_ignoring_last_label

model.compile(optimizer = optimizer, loss = loss_fn , metrics =[UpdatedMeanIoU(num_classes=5)])#,sample_weight_mode='temporal')#UpdatedMeanIoU(num_classes=5)#tf.keras.metrics.SparseCategoricalAccuracy()#MyMeanIoU(num_classes=5)#loss_weights=loss_weights#[tf.keras.metrics.SparseCategoricalAccuracy()]#[tf.keras.metrics.MeanIoU(num_classes=5)])#['accuracy'])#[sparse_accuracy_ignoring_last_label])#,sample_weight_mode='temporal')

# checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}
# callbacks.append(checkpoint)
### AVVIO IL TRAINING #####
model.summary()
# history = 
model.fit(x = x_train,steps_per_epoch=steps,epochs=EPOCHS,validation_data=x_validation)#,callbacks=[callbacks])#, callbacks=[cp_callback])#,callbacks=[callbacks])
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