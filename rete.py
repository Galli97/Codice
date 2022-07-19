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
    x = Dropout(0.75)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.75)(x)
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

def rete_vgg16_dilation(img_size=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=5):
    
    
    vggmodel = tf.keras.applications.vgg16.VGG16(input_shape=img_size, weights='imagenet',include_top=False)

    vggmodel = Sequential(vggmodel.layers[:-8])
    for layer in vggmodel.layers:        
        layer._trainable = False
    for i, layer in enumerate(vggmodel.layers):
        layer._name = 'layer_' + str(i)
    x = vggmodel.output
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    
    x = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.75)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.75)(x)
    x = Conv2D(classes, (3, 3), activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = tf.keras.layers.UpSampling2D(32,interpolation='bilinear')(x)
    
    x = Activation('softmax')(x)
    model = Model(inputs=vggmodel.input, outputs=x)
    weights_path = os.path.expanduser('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.load_weights(weights_path, by_name=True)
    return model


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def conv_block(input, num_filters):
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
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape,weight_decay=0.,classes=5):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    vgg16 = Sequential(vgg16.layers[:-4])
    for layer in vgg16.layers:        
        layer.trainable = False
    x = vgg16.output
    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output 

    """ Bridge """
    x = vgg16.output
    
    # b1 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    # b1 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv2', kernel_regularizer=l2(weight_decay))(b1)
    # b1 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block4_conv3', kernel_regularizer=l2(weight_decay))(b1)         ## (32 x 32)
    # s4 = b1 ## (64 x 64)
    # b1_pooling = MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block5_pool')(b1)
  
    b2 = Conv2D(1024, (3, 3), activation='relu', padding='same',dilation_rate=(12,12), name='fc5', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    b2 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc6', kernel_regularizer=l2(weight_decay))(b2)
    x = Dropout(0.5)(x)
    #b3 = Conv2D(classes, (3, 3), activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(b2)
    #x = tf.keras.layers.UpSampling2D(32,interpolation='bilinear')(x)
    
    """ Decoder """
           
    d1 = decoder_block(b2, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(5, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model

# if __name__ == "__main__":
#     input_shape = (512, 512, 3)
#     model = build_vgg16_unet(input_shape)
#     model.summary()

