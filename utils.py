import numpy as np
import tensorflow as tf
import random
import cv2
import tensorflow as tf
from keras.layers import *
import keras.backend as K


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
        tmp1 = np.load(f)
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
    i = random.randint(1,5)
    i=i/10
    f = random.randint(5,10)
    f=f/10
    contrasted = tf.image.stateless_random_contrast(image, i, f)
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
    A = random.randint(int(fix*20/110),fix)
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
        choose = random.randint(1,3)
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
            new_image = grayscale(image)
            new_label = label
        elif(choose == 5):
            new_image = saturate(image)
            new_label = label
        elif(choose == 6):
            new_image = contrast(image)
            new_label = label
        # elif(choose == 4):
        #     new_image = translation(image)
        #     new_label = translation(label)
        tmp1a.append(new_image)#[i]=new_image
        tmp2a.append(new_label)#[i]=new_label
    A=A-skipped
    return tmp1a,tmp2a,A


def decode_masks(tmp2):
    soil=4;
    bedrock=1;
    sand=2;
    bigrock=3;
    null=0;
    decoded_images = np.empty((len(tmp2), 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]
      #reduct_label = label[:,:,0]                        
      image = np.empty((64, 64, 3), dtype=np.uint8) 
      #image[:,:,0]=reduct_label  
      for i in range(0,64):
          for j in range(0,64): 
              channels_xy = label[i,j];          #SOIL is kept black, NULL (no label) is white 
              if channels_xy==bedrock:      #BEDROCK --->RED
                  image[i,j,0]=255
                  image[i,j,1]=0
                  image[i,j,2]=0
              elif channels_xy==sand:    #SAND --->GREEN
                  image[i,j,0]=0
                  image[i,j,1]=255
                  image[i,j,2]=0
              elif channels_xy==bigrock:    #BIG ROCK ---> BLUE
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


def decode_predictions(tmp2):
    decoded_images = np.empty((len(tmp2), 64, 64, 1), dtype=np.uint8)  #Qui ho N immagini
    for n in range (len(tmp2)):
      label = tmp2[n]                   
      image = np.empty((64, 64, 1), dtype=np.uint8) 
      for i in range(0,64):
          for j in range(0,64): 
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
    
    null_pixels = 1667361
    bedrock_pixels =  2143243
    sand_pixels =  1553591
    bigrock_pixels = 750886
    soil_pixels = 1327351
    

    PIXELS=soil_pixels+bedrock_pixels + sand_pixels+bigrock_pixels#+null_pixels ;
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([0,PIXELS/bedrock_pixels,PIXELS/sand_pixels,PIXELS/bigrock_pixels,PIXELS/soil_pixels])
    #class_weights = tf.constant([1,1,1,1,1])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights
########################
def add_sample_weights_val(image, label):
    

    null_pixels = 388903  
    bedrock_pixels =  685710 
    sand_pixels =  514451 
    bigrock_pixels =  110021 
    soil_pixels = 164595 
    

    PIXELS=soil_pixels+bedrock_pixels + sand_pixels+bigrock_pixels#+null_pixels ;
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([0, PIXELS/bedrock_pixels,PIXELS/sand_pixels,PIXELS/bigrock_pixels,PIXELS/soil_pixels])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    

    return image, label, sample_weights

        