import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model,Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import image
import cv2
import numpy as np
from keras.regularizers import l2
import keras.backend as K
import os
from keras.utils.data_utils import get_file
from utils import *
### QUESTA FUNZIONE RECUPERA I PESI DELLA RESNET50
def get_weights_path_resnet():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path




class BilinearInitializer(keras.initializers.Initializer):
    '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
    def __call__(self, shape, dtype=None, **kwargs):
        kernel_size, _, filters, _ = shape
        arr = np.zeros((kernel_size, kernel_size, filters, filters))
        ## make filter that performs bilinear interpolation through Conv2DTranspose
        upscale_factor = (kernel_size+1)//2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
                 (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
        for i in range(filters):
            arr[..., i, i] = kernel
        return tf.convert_to_tensor(arr, dtype=dtype)




##### IN QUESTA RETE SEGUO LA STRUTTURA DEL PAPER #############
def rete(input_shape=None, weight_decay=0., batch_shape=None, classes=5):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # I1 = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same',name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2),name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3),  kernel_initializer='normal',dilation_rate=(2,2),activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    #x = Conv2D(classes, (1, 1),kernel_initializer='he_normal', activation='linear', padding='same', kernel_regularizer=l2(weight_decay))(x)
    
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(16,16), strides=(16,16),
    #                                  padding='same', use_bias=False,
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='trans')(x)
    x = Conv2D(classes, 1,strides=(1, 1), activation='softmax', padding='valid',kernel_regularizer=l2(weight_decay))(x)
    
    #x = Activation('softmax')(x)
   
    model = Model(img_input, x)
    # for layer in model.layers[:-11]:        
    #     layer.trainable = False
    #     print(layer.name)

    weights_path = os.path.expanduser('/content/drive/MyDrive/Tesi/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    #weights_path = os.path.expanduser('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.load_weights(weights_path,by_name=True)

    # checkpoint_path = "Checkpoint/cp.ckpt"
    # model.load_weights(checkpoint_path)

    return model

##############################################################


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
def get_weights_path_vgg16():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path


##############################################################
def rete_Resnet101(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    model_input = keras.Input(shape=(img_size, img_size, 3))

    res_model = tf.keras.applications.resnet.ResNet101(weights='imagenet',include_top=False,input_tensor=model_input)

    #res_model = Sequential(res_model.layers[:-4])
    # for layer in res_model.layers: #[:-4]:        
    #     layer.trainable = False
    #     print(layer.name)
    #x = res_model.get_layer("conv4_block6_2_relu").output
    x = res_model.output
    
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3),  kernel_initializer='normal',dilation_rate=(2,2),activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    #x = Conv2D(classes, (1, 1),kernel_initializer='he_normal', activation='linear', padding='same', kernel_regularizer=l2(weight_decay))(x)
    
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(16,16), strides=(16,16),
    #                                  padding='same', use_bias=False,
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='fc3')(x)
    x = Conv2D(classes, 1,strides=(1, 1), activation='softmax', padding='valid',kernel_regularizer=l2(weight_decay))(x)
    
   
    # x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs=res_model.input, outputs=x)

    return model

##############################################################

def rete_Res50(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    model_input = keras.Input(shape=(img_size, img_size, 3))

    res_model = tf.keras.applications.resnet.ResNet50(weights='imagenet',include_top=False,input_tensor=model_input)

    #res_model = Sequential(res_model.layers[:-4])
    # for layer in res_model.layers: #[:-4]:        
    #     layer.trainable = False
    #     print(layer.name)
    #x = res_model.output
    x = res_model.get_layer("conv4_block6_2_relu").output

    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3),  kernel_initializer='normal',dilation_rate=(2,2),activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    #x = Conv2D(classes, (1, 1),kernel_initializer='he_normal', activation='linear', padding='same', kernel_regularizer=l2(weight_decay))(x)
    
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(16,16), strides=(16,16),
    #                                  padding='same', use_bias=False,
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='fc3')(x)
    x = Conv2D(classes, 1,strides=(1, 1), activation='softmax', padding='valid',kernel_regularizer=l2(weight_decay))(x)
    
   
    # x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs=res_model.input, outputs=x)

    return model

##############################################################
##############################################################
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def conv_block_g(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_g(x, num_filters)
    return x

def build_vgg16_unet(input_shape,weight_decay=0.,classes=5):
    """ Input """
    inputs = Input(input_shape)

    # """ Pre-trained VGG16 Model """
    # vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    # vgg16 = Sequential(vgg16.layers[:-8])
    # # for layer in vgg16.layers:        
    # #     layer.trainable = False
    # #x = vgg16.output
    # """ Encoder """
    # s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    # s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    # s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    # #s4 = vgg16.get_layer("block4_conv3").output 

    # """ Bridge """
    # x = vgg16.output
     # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    s1=x #128
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    s2=x #64
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    s3=x #32
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same',name='block3_pool')(x)

    b1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',dilation_rate=(2,2), kernel_regularizer=l2(weight_decay))(x)
    b1 = BatchNormalization()(b1)
    b1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',dilation_rate=(2,2), kernel_regularizer=l2(weight_decay))(b1)
    b1 = BatchNormalization()(b1)
    b1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',dilation_rate=(2,2), kernel_regularizer=l2(weight_decay))(b1) 
    b1 = BatchNormalization()(b1)       
    s4 = b1 #16
    b1_pooling = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(b1)
  
    b2 = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc5', kernel_regularizer=l2(weight_decay))(b1_pooling)
    b2 = BatchNormalization()(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc6', kernel_regularizer=l2(weight_decay))(b2)
    b2 = BatchNormalization()(b2)
    b2 = Dropout(0.5)(b2)
    b3 = Conv2D(classes, (3, 3),  kernel_initializer='normal',dilation_rate=(2,2), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(b2)
    b3 = BatchNormalization()(b3)
   
    d1 = decoder_block(b3, s4, 128)                
    d2 = decoder_block(d1, s3, 64)                     
    d3 = decoder_block(d2, s2, 32)                     
    d4 = decoder_block(d3, s1, 16)                    

    """ Output """
    outputs = Conv2D(5, 1, padding="valid", activation="softmax",kernel_regularizer=l2(weight_decay))(d4)
    
    model = Model(inputs, outputs, name="VGG16_Skip")
    weights_path = os.path.expanduser('/content/drive/MyDrive/Tesi/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    #weights_path = os.path.expanduser('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.load_weights(weights_path,by_name=True)
    return model
##############################################################

##############################################################


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    # x = atrous_conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)

    

    # x = atrous_conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay,atrous_rate=(10, 10), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = atrous_identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay,atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    
    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # x = identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    # x = identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    #classifying layer
    # x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Conv2D(5, (3, 3),  kernel_initializer='normal',dilation_rate=(2,2),activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    #x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(16,16), strides=(16,16),
    #                                  padding='same', use_bias=False,
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='trans')(x)
    x = Conv2D(classes, 1,strides=(1, 1), activation='softmax', padding='valid',kernel_regularizer=l2(weight_decay))(x)
    
   # x = Activation('softmax')(x)

    
    model = Model(img_input, x)
    weights_path = get_weights_path_resnet()#os.path.expanduser('/content/drive/MyDrive/Tesi/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights(weights_path,by_name=True)

    return model

    ##############################################################


#########################

def mobile(shape=None,weight_decay=0.0005):
    mob = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=shape,include_top=False,weights='imagenet',classes=5,classifier_activation=None)
    x=mob.output
    #x=mob.get_layer('out_relu').output

    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(10,10), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(5, (3, 3),  kernel_initializer='normal', dilation_rate=(2,2),activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    
    # x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(32,32), strides=(16,16),
    #                                  padding='same', use_bias=False,
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='trans')(x)
    x = tf.keras.layers.UpSampling2D(32,interpolation='bilinear')(x)
    output = Conv2D(5, 1, padding="valid", activation="softmax",kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=mob.input, outputs=output)
    return model

##############################################################
def rete_vgg16(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    vggmodel = tf.keras.applications.vgg16.VGG16(input_shape=img_size, weights='imagenet',include_top=False)

    vggmodel = Sequential(vggmodel.layers[:-4])
    for layer in vggmodel.layers:#[:-4]:        
        layer.trainable = False
    x = vggmodel.output
   
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3), activation='linear',kernel_initializer='he_normal', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = keras.layers.Conv2DTranspose(filters=classes, kernel_size=(16,16), strides=(16,16),
                                     padding='same', use_bias=False, activation='softmax',
                                     kernel_initializer=BilinearInitializer(),
                                     kernel_regularizer=l2(weight_decay),
                                     name='fc3')(x)
    # x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs=vggmodel.input, outputs=x)

    return model
##############################################################

##############################################################
def rete_vgg16_dilation(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    
    vggmodel = tf.keras.applications.vgg16.VGG16(input_shape=img_size, weights='imagenet',include_top=False)

    vggmodel = Sequential(vggmodel.layers[:-8])
    for layer in vggmodel.layers:        
        layer._trainable = False
    # for i, layer in enumerate(vggmodel.layers):
    #     layer._name = 'layer_' + str(i)
    x = vggmodel.output
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.75)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.75)(x)
    x = Conv2D(classes, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    
    x = Conv2D(classes, 1, activation='softmax',padding='valid',kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=vggmodel.input, outputs=x)
    # weights_path = os.path.expanduser('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # model.load_weights(weights_path, by_name=True)
    
    return model
##############################################################


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    weight_decay=0.0005
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        kernel_regularizer=l2(weight_decay)
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes,weight_decay):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    # x = keras.layers.Conv2DTranspose(filters=5, kernel_size=((image_size // x.shape[1], image_size // x.shape[2])), strides=((image_size // x.shape[1], image_size // x.shape[2])),
    #                                  padding='same', use_bias=False, activation='softmax',
    #                                  kernel_initializer=BilinearInitializer(),
    #                                  kernel_regularizer=l2(weight_decay),
    #                                  name='trans')(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation='softmax')(x)
    return keras.Model(inputs=model_input, outputs=model_output)

    ##################
    import tensorflow as tf
#import all necessary layers
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
###########################
# MobileNet block
def mobilnet_block (x, filters, strides,weight_decay):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1,kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def mobilnet_block_dilation (x, filters, strides,weight_decay,dilation):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same',kernel_regularizer=l2(weight_decay),dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1,kernel_regularizer=l2(weight_decay),dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def mobileNET(shape = (224,224,3),weight_decay=0.0005): 
    input = Input(shape)
    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same',kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # main part of the model
    x = mobilnet_block(x, filters = 64, strides = 1,weight_decay=weight_decay)
    x = mobilnet_block(x, filters = 128, strides = 2,weight_decay=weight_decay)
    x = mobilnet_block(x, filters = 128, strides = 1,weight_decay=weight_decay)
    x = mobilnet_block(x, filters = 256, strides = 2,weight_decay=weight_decay)
    x = mobilnet_block(x, filters = 256, strides = 1,weight_decay=weight_decay)
    x = mobilnet_block(x, filters = 512, strides = 2,weight_decay=weight_decay)
    for _ in range (5):
        x = mobilnet_block_dilation(x, filters = 512, strides = 1,weight_decay=weight_decay,dilation=(2,2))
    x = mobilnet_block_dilation(x, filters = 1024, strides = 2,weight_decay=weight_decay,dilation=(10,10))
    x = mobilnet_block(x, filters = 1024, strides = 1,weight_decay=weight_decay)
    x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_last')(x)

    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc4', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(5, (3, 3),  kernel_initializer='normal', activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    

    x = tf.keras.layers.UpSampling2D(64,interpolation='bilinear')(x)
    output = Conv2D(5, 1, padding="valid", activation="softmax",kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=input, outputs=output)
    #model.summary()
    weights_path = os.path.expanduser('/content/drive/MyDrive/Tesi/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5')
    #weights_path = os.path.expanduser('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.load_weights(weights_path,by_name=True)
    return model