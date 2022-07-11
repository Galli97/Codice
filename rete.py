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

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# class UpSampling2D(Layer):
#     def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
#         if data_format == 'default':
#             data_format = K.image_data_format()
#         self.size = tuple(size)
#         if target_size is not None:
#             self.target_size = tuple(target_size)
#         else:
#             self.target_size = None
#         assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
#         self.data_format = data_format
#         self.input_spec = [InputSpec(ndim=4)]
#         super(UpSampling2D, self).__init__(**kwargs)


##### IN QUESTA RETE SEGUO LA STRUTTURA DEL PAPER ###
def rete(input_shape=None, weight_decay=0., batch_shape=None, classes=5):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # I1 = Input(input_shape)

    # model = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=img_input, pooling=None)
    # model.layers.pop()
    # # model.outputs = [model.layers[-1].output]
    # # model.layers[-1]._outbound_nodes = []

    # for layer in model.layers:
    #     layer._name = layer.name
    #     layer._trainable = False
    # x = model.output

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    #x = tf.keras.layers.BatchNormalization()(x)##########
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    #x = tf.keras.layers.BatchNormalization()(x)##########
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    #x = tf.keras.layers.BatchNormalization()(x)##########
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same',name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    #x = tf.keras.layers.BatchNormalization()(x)##########
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    #x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    #x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3), activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)

    # x = upsample(512, 3)(x) # 4x4 -> 8x8
    # x = upsample(256, 3)(x)  # 8x8 -> 16x16
    # x = upsample(128, 3)(x)  # 16x16 -> 32x32
    # #x = upsample(64, 3)(x)  # 32x32 -> 64x64
    # x = tf.keras.layers.Conv2DTranspose(
    #   filters=classes, kernel_size=3, strides=2,
    #   padding='same') (x)
    # img_size=input_shape[0];
    # x = layers.UpSampling2D(
    #     size=(img_size // x.shape[1], img_size // x.shape[2]),
    #     interpolation="bilinear",
    # )(x)
    #x = BilinearUpSampling2D(target_size=tuple(image_size))(x)
    #x = tf.keras.layers.Reshape((64*64,5))(x)
    x = Activation('softmax')(x)
    #x = tf.keras.layers.Reshape((64,64,1))(x)
    model = Model(img_input, x)

    weights_path = get_weights_path_resnet()
    model.load_weights(weights_path, by_name=True)
    return model


##### COME RETE 2 MA MODELLO SEQUENTIAL E AGGIUNTA DI BATCH NORMALIZATION, DROPOUT E AVERAGEPOOLING ####
def rete_2(input_shape=None, weight_decay=0., batch_shape=None, classes=5):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # I1 = Input(input_shape)
    
    # tl_model = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=img_input, pooling=None)
    # tl_model.layers.pop()
    # # tl_model.outputs = [tl_model.layers[-1].output]
    # # tl_model.layers[-1]._outbound_nodes = []

    # for layer in tl_model.layers:
    #     layer._name = layer.name
    #     layer._trainable = False
    
    model=keras.Sequential()
    model.add(img_input)
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())##########
    model.add(AveragePooling2D((2, 2), strides=(2, 2),padding='same', name='block1_pool'))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())##########
    model.add(AveragePooling2D((2, 2), strides=(2, 2),padding='same', name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())##########  
    model.add(AveragePooling2D((2, 2), strides=(2, 2),padding='same',name='block3_pool'))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=2, name='block4_conv1', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=2, name='block4_conv2', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=2, name='block4_conv3', kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())##########
    model.add(AveragePooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool'))

    # Block 5
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=12, name='block5_conv1', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(0.5))      #############
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(0.5))  #########
    model.add(Conv2D(classes, (3, 3), activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay)))
    
    model.add(UpSampling2D(16))

    # img_size=input_shape[0];
    # x = layers.UpSampling2D(
    #     size=(img_size // x.shape[1], img_size // x.shape[2]),
    #     interpolation="bilinear",
    # )(x)

    #x = tf.keras.layers.Reshape((64*64,5))(x)
    model.add(Activation('softmax'))
  

    #model = Model(img_input, x)
    model.built = True
    # weights_path = get_weights_path_resnet()
    # model.load_weights(weights_path, by_name=True)
    return model


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
def get_weights_path_vgg16():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path


def rete_vgg16(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    vggmodel = tf.keras.applications.vgg16.VGG16(input_shape=img_size, weights='imagenet',include_top=False)

    vggmodel = Sequential(vggmodel.layers[:-4])
    for layer in vggmodel.layers:#[:-4]:        
        layer.trainable = False
    x = vggmodel.output
   
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc1', kernel_regularizer=l2(weight_decay))(x)
    #x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    #x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3), activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.UpSampling2D(16,interpolation='bilinear')(x)
    x = Activation('softmax')(x)
    model = Model(inputs=vggmodel.input, outputs=x)

    return model

def rete_vgg16_dilation(image_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    
    vggmodel = tf.keras.applications.vgg16.VGG16(input_shape=image_size, weights='imagenet',include_top=False)

    vggmodel = Sequential(vggmodel.layers[:-8])
    for layer in vggmodel.layers:        
        layer.trainable = False
    x = vggmodel.output
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(classes, (3, 3), activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.UpSampling2D(32,interpolation='bilinear')(x)
    
    x = Activation('softmax')(x)
    model = Model(inputs=vggmodel.input, outputs=x)

    # weights_path = get_weights_path_vgg16()
    # model.load_weights(weights_path, by_name=True)

    return model

############ DEEPLABV3+ IMPLEMENTATA NELL'ESEMPIO DI TENSORFLOW#######
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
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

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
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
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)
