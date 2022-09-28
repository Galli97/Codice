import numpy as np
import tensorflow as tf
import random
import cv2
import tensorflow as tf
from keras.layers import *
import keras.backend as K
import keras
import sklearn.metrics 
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     #print("Normalized confusion matrix")
    # else:
    #     pass
    #     #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def save_np_arrays(tmp1):
    with open('image_arrays.npy','wb') as f:
        np.save(f,tmp1)

def save_sparse_np_arrays(tmp1):
    with open('image_arrays_sparse.npy','wb') as f:
        np.save(f,tmp1)

def save_patches(tmp1):
    with open('image_patches.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_images(tmp1):
    with open('cropped_images.npy','wb') as f:
        np.save(f,tmp1)

def save_patches_TEST(tmp1):
    with open('image_patches_TEST.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_images_TEST(tmp1):
    with open('cropped_images_TEST.npy','wb') as f:
        np.save(f,tmp1)

def save_predictions(tmp1):
    with open('predictions.npy','wb') as f:
        np.save(f,tmp1)

def save_final_images(tmp1):
    with open('final_images.npy','wb') as f:
        np.save(f,tmp1)

def get_np_arrays(file):
    with open(file,'rb') as f:
        tmp1 = np.load(f,allow_pickle=True)
    return tmp1

def save_np_arrays_labels(tmp2):
    with open('label_arrays.npy','wb') as f:
        np.save(f,tmp2)
def save_sparse_np_arrays_labels(tmp2):
    with open('label_arrays_sparse.npy','wb') as f:
        np.save(f,tmp2)
        
def save_label_patches(tmp1):
    with open('label_patches.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_labels(tmp1):
    with open('cropped_labels.npy','wb') as f:
        np.save(f,tmp1)

def save_label_patches_TEST(tmp1):
    with open('label_patches_TEST.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_labels_TEST(tmp1):
    with open('cropped_labels_TEST.npy','wb') as f:
        np.save(f,tmp1)

def save_final_labels(tmp1):
    with open('final_labels.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_images512(tmp1):
    with open('cropped_images512.npy','wb') as f:
        np.save(f,tmp1)

def save_cropped_labels512(tmp1):
    with open('cropped_labels512.npy','wb') as f:
        np.save(f,tmp1)
def save_label_patches512(tmp1):
    with open('label_patches512.npy','wb') as f:
        np.save(f,tmp1)
def save_patches512(tmp1):
    with open('image_patches512.npy','wb') as f:
        np.save(f,tmp1)
### datagenerator prepara i dati per il training
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

def flip(image):
    flipped = tf.image.flip_left_right(image)
    return flipped
def grayscale(image):
    grayscaled = tf.image.rgb_to_grayscale(image)
    return grayscaled

def saturate(image):
    i = random.randint(1,10)
    saturated = tf.image.adjust_saturation(image, i)
    return saturated

def brightness(image):
    i = random.randint(1,10)
    i=i/10
    bright = tf.image.adjust_brightness(image, i)
    return bright 


def contrast(image):
    i = random.randint(1,4)
    i=i/10
    f = random.randint(5,10)
    f=f/10
    seed = (1, 2)
    contrasted = tf.image.stateless_random_contrast(image,i,f,seed)
    return contrasted

def cropp(image,central_fraction):
    cropped = tf.image.central_crop(image, central_fraction=0.5)
    return cropped

def rotate(image):
    rotated = tf.image.rot90(image)
    return rotated

def translation(image):
  height, width = image.shape[:2]
  quarter_height, quarter_width = height / 4, width / 4
  T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
  img_translation = cv2.warpAffine(image, T, (width, height))
  return img_translation  

def augment(image_list,label_list,N):
    fix=N-1        #voglio lavorare solo sulle immagini della lista iniziale
    A = random.randint(int(fix*50/100),fix)
    skipped=0
    indices=[]
    tmp1a = []# np.empty((A, 64, 64, 1), dtype=np.uint8)  #Qui ho A immagini
    tmp2a = []#np.empty((A, 64, 64, 1), dtype=np.uint8) 
    for i in range (0,A):
        a = random.randint(0,fix)
        if a in indices:
          skipped+=1
          continue  
        image=image_list[a] 
        label=label_list[a]
        indices.append(a)
        choose = random.randint(1,5)
        #print(a)
        if(choose == 1):
            new_image = rotate(image)
            new_label = rotate(label)
        elif(choose == 2):
            new_image = brightness(image)
            new_label = label
        elif(choose == 3):
            new_image = flip(image)
            new_label = flip(label)
        elif(choose == 4):
            new_image = saturate(image)
            new_label = label
        elif(choose == 5):
            new_image = contrast(image)
            new_label = label
        # elif(choose == 4):
        #     new_image = translation(image)
        #     new_label = translation(label)
        tmp1a.append(new_image)#[i]=new_image
        tmp2a.append(new_label)#[i]=new_label
    A=A-skipped
    return tmp1a,tmp2a,A

def decode_masks_Notsparse(tmp2,SHAPE):
    soil=[0,0,0,0,1];
    bedrock=[0,1,0,0,0];
    sand=[0,0,1,0,0];
    bigrock=[0,0,0,1,0];
    null=[1,0,0,0,0];
    
    decoded_images = np.empty((len(tmp2), SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]
      #reduct_label = label[:,:,0]                        
      image = np.empty((SHAPE, SHAPE, 3), dtype=np.uint8) 
      #image[:,:,0]=reduct_label  
      for i in range(0,SHAPE):
          for j in range(0,SHAPE): 
              channels_xy = label[i,j];          #SOIL is kept black, NULL (no label) is white 
              if all(channels_xy==bedrock):      #BEDROCK --->RED
                  image[i,j,0]=255
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif all(channels_xy==sand):    #SAND --->GREEN
                  image[i,j,0]=0
                  image[i,j,1]=255
                  image[i,j,2]=0
              elif all(channels_xy==bigrock):    #BIG ROCK ---> BLUE
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=255
              elif all(channels_xy==soil):    #SOIL ---> BLACK
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif all(channels_xy==null):    #NULL ---> WHITE
                  image[i,j,0]=255
                  image[i,j,1]=255
                  image[i,j,2]=255
              decoded_images[n]=image
    return decoded_images

def decode_null(tmp2,SHAPE):
    soil=0;
    bedrock=1;
    sand=2;
    bigrock=3;
    null=255;
    
    decoded_images = np.empty((len(tmp2), SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
    
    label = tmp2
    #reduct_label = label[:,:,0]                        
    image = np.empty((SHAPE, SHAPE, 3), dtype=np.uint8) 
    #image[:,:,0]=reduct_label  
    for i in range(0,SHAPE):
        for j in range(0,SHAPE): 
            channels_xy = label[i,j];          #SOIL is kept black, NULL (no label) is white 
            if channels_xy[0]==bedrock:      #BEDROCK --->BLUE
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=0
            elif channels_xy[0]==sand:    #SAND --->GREEN
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=0
            elif channels_xy[0]==bigrock:    #BIG ROCK ---> RED
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=0
            elif channels_xy[0]==soil:    #SOIL ---> BLACK
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=0
            elif channels_xy[0]==null:    #NULL ---> WHITE
                image[i,j,0]=255
                image[i,j,1]=255
                image[i,j,2]=255
            decoded_images=image
    return decoded_images

def New_label(img1,img2,SHAPE):
    
    soil=0;
    null= 255;
                            
    image = np.empty((SHAPE, SHAPE, 3), dtype=np.uint8) 
    #image[:,:,0]=reduct_label  
    for i in range(0,SHAPE):
        for j in range(0,SHAPE): 
            channels_xy = img1[i,j];          #SOIL is kept black, NULL (no label) is white 
            if channels_xy[0]==soil:    #SOIL ---> BLACK
                image[i,j] = img2[i,j]
            elif channels_xy[0]==null:    #NULL ---> WHITE
                image[i,j,0]=255
                image[i,j,1]=255
                image[i,j,2]=255
    return image
    
def decode_masks(tmp2,SHAPE):
    soil=4;
    bedrock=1;
    sand=2;
    bigrock=3;
    null=0;
    
    decoded_images = np.empty((len(tmp2), SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]
      #reduct_label = label[:,:,0]                        
      image = np.empty((SHAPE, SHAPE, 3), dtype=np.uint8) 
      #image[:,:,0]=reduct_label  
      for i in range(0,SHAPE):
          for j in range(0,SHAPE): 
              channels_xy = label[i,j];          #SOIL is kept black, NULL (no label) is white 
              if channels_xy==bedrock:      #BEDROCK --->BLUE
                  image[i,j,0]=255
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif channels_xy==sand:    #SAND --->GREEN
                  image[i,j,0]=0
                  image[i,j,1]=255
                  image[i,j,2]=0
              elif channels_xy==bigrock:    #BIG ROCK ---> RED
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=255
              elif channels_xy==soil:    #SOIL ---> BLACK
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif channels_xy==null:    #NULL ---> WHITE
                  image[i,j,0]=255
                  image[i,j,1]=255
                  image[i,j,2]=255
              decoded_images[n]=image
    return decoded_images

def decode_labels_overlay(tmp2,SHAPE):
    soil=0;
    bedrock=1;
    sand=2;
    bigrock=3;
    null=255;
    
    decoded_images = np.empty((len(tmp2), SHAPE, SHAPE, 3), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]
      #reduct_label = label[:,:,0]                        
      image = np.empty((SHAPE, SHAPE, 3), dtype=np.uint8) 
      #image[:,:,0]=reduct_label  
      for i in range(0,SHAPE):
          for j in range(0,SHAPE): 
              channels_xy = label[i,j];          #SOIL is kept black, NULL (no label) is white 
              if channels_xy==bedrock:      #BEDROCK --->BLUE
                  image[i,j,0]=255
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif channels_xy==sand:    #SAND --->GREEN
                  image[i,j,0]=0
                  image[i,j,1]=255
                  image[i,j,2]=0
              elif channels_xy==bigrock:    #BIG ROCK ---> RED
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=255
              elif channels_xy==soil:    #SOIL ---> BLACK
                  image[i,j,0]=0
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif channels_xy==null:    #NULL ---> WHITE
                  image[i,j,0]=255
                  image[i,j,1]=255
                  image[i,j,2]=255
              decoded_images[n]=image
    return decoded_images


def decode_predictions(tmp2,SHAPE):
    decoded_images = np.empty((len(tmp2), SHAPE, SHAPE, 1), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]                   
      image = np.empty((SHAPE, SHAPE, 1), dtype=np.uint8) 
      for i in range(0,SHAPE):
          for j in range(0,SHAPE): 
              image[i,j,0]=np.argmax(label[i,j])
              decoded_images[n]=image
    return decoded_images
##########################



def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


##############FUNZIONA################
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)
#########################################
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

def dice_coef_func(smooth=1, threshold=0.5):
    def dice_coef(y_true, y_pred):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        prediction = tf.where(y_pred > threshold, 1, 0)
        prediction = tf.cast(prediction, dtype=y_true.dtype)
        ground_truth_area = tf.reduce_sum(
            y_true, axis=(1, 2, 3))
        prediction_area = tf.reduce_sum(
            prediction, axis=(1, 2, 3))
        intersection_area = tf.reduce_sum(
            y_true*y_pred, axis=(1, 2, 3))
        combined_area = ground_truth_area + prediction_area
        dice = tf.reduce_mean(
            (2*intersection_area + smooth)/(combined_area + smooth))
        return dice
    return dice_coef

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_f = tf.cast(y_pred_f, dtype=y_true_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def iou_coef(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  #y_pred = tf.math.argmax(y_pred)
  intersection = K.sum(y_true * y_pred)
  union = K.sum(y_true)+K.sum(y_pred)-intersection
  iou = intersection / union 
  return iou



def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.cast(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), tf.float32)) / K.sum(tf.cast(legal_labels,tf.float32))


########################################################
#######################################################
  
def add_sample_weights(image, label):
    
    #128 Res512 crop128
    null_pixels = 9988015  
    bedrock_pixels =  7457808 
    sand_pixels =  6429031 
    bigrock_pixels =  1928322 
    soil_pixels = 3507800 
 
    #128 Res256 crop128
    # null_pixels = 6229515  
    # bedrock_pixels =  8523534 
    # sand_pixels =  7327922 
    # bigrock_pixels =  747234 
    # soil_pixels = 1420115 

    #128 Dataset_resize
    # null_pixels = 16932880  
    # bedrock_pixels =  14266903 
    # sand_pixels =  7908483 
    # bigrock_pixels =  709744 
    # soil_pixels = 2206950  

    #128 Dataset_BigRock
    # null_pixels = 9886938  
    # bedrock_pixels =  7087121 
    # sand_pixels =  6179368 
    # bigrock_pixels =  7407219 
    # soil_pixels = 5942906  

    #128 Dataset_1 255
    # null_pixels = 7438103  
    # bedrock_pixels =  7015639 
    # sand_pixels =  6191210 
    # bigrock_pixels =  3382388 
    # soil_pixels = 5529396  

    #128 Dataset_bigRock 2
    null_pixels = 6888445  
    bedrock_pixels =  5067169 
    sand_pixels =  4292427 
    bigrock_pixels =  4942703 
    soil_pixels = 3975080  


    PIXELS=soil_pixels+bedrock_pixels + sand_pixels+bigrock_pixels#+null_pixels ;
    #perc_null=1-null_pixels/PIXELS
    perc_bedrock=1-bedrock_pixels/PIXELS
    perc_sand=1-sand_pixels/PIXELS
    perc_bigrock=1-bigrock_pixels/PIXELS
    perc_soil=1-soil_pixels/PIXELS

    class_weights = tf.constant([0,perc_bedrock,perc_sand,perc_bigrock,perc_soil])
    #class_weights = tf.constant([0,PIXELS/bedrock_pixels,PIXELS/sand_pixels,PIXELS/bigrock_pixels,PIXELS/soil_pixels])
    #class_weights = tf.constant([1,1,1,1,1])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights

######RESNET########
from keras.layers import *
from keras.layers.merge import Add
from keras.regularizers import l2

# The original help functions from keras does not have weight regularizers, so I modified them.
# Also, I changed these two functions into functional style
def identity_block(kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f

# Atrous-Convolution version of residual blocks
def atrous_identity_block(kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

def atrous_conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f