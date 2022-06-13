import numpy as np
import tensorflow as tf
import random
import cv2
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils



def save_np_arrays(tmp1):
    with open('image_arrays.npy','wb') as f:
        np.save(f,tmp1)

def save_sparse_np_arrays(tmp1):
    with open('image_arrays_sparse.npy','wb') as f:
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

def saturate(image,i=3):
    saturated = tf.image.adjust_saturation(image, i)
    return saturated

def brightness(image,i=0.4):
    bright = tf.image.adjust_brightness(image, i)
    return bright 

def cropp(image,central_fraction):
    cropped = tf.image.central_crop(image, central_fraction=0.5)
    return cropped

def rotate(image):
    rotated = tf.image.rot90(image)
    return rotated

def preprocess(image_list,label_list,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               is_training=True,):
    # The probability of flipping the images and labels
    # left-right during training
    _PROB_OF_FLIP = 0.5

    processed_image = tf.cast(image_list, tf.float32)
    label = tf.cast(label_list, tf.int32)
     # Resize image and label to the desired range.
    if min_resize_value or max_resize_value:
        [processed_image, label] = (
            preprocess_utils.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        # The `original_image` becomes the resized image.
        original_image = tf.identity(processed_image)
        # Data augmentation by randomly scaling the inputs.
        if is_training:
            scale = preprocess_utils.get_random_scale(
                min_scale_factor, max_scale_factor, scale_factor_step_size)
            processed_image, label = preprocess_utils.randomly_scale_image_and_label(
                processed_image, label, scale)
            processed_image.set_shape([None, None, 3])
          # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = tf.shape(processed_image)
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + tf.maximum(crop_height - image_height, 0)
        target_width = image_width + tf.maximum(crop_width - image_width, 0)

        # Randomly crop the image and label.
        if is_training and label is not None:
            processed_image, label = preprocess_utils.random_crop(
                [processed_image, label], crop_height, crop_width)

        processed_image.set_shape([crop_height, crop_width, 3])

        if label is not None:
            label.set_shape([crop_height, crop_width, 1])

        if is_training:
            # Randomly left-right flip the image and label.
            processed_image, label, _ = preprocess_utils.flip_dim(
                [processed_image, label], _PROB_OF_FLIP, dim=1)

        return original_image, processed_image, label
#     fix=len(image_list)-1        #voglio lavorare solo sulle immagini della lista iniziale
#     A = random.randint(10,fix)
#     tmp1a = np.empty((A, 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
#     tmp2a = np.empty((A, 64, 64, 3), dtype=np.uint8) 
#     for i in range (0,A):
#         a = random.randint(0,fix)
#         image = cv2.imread(image_list[a])[:,:,[2,1,0]]
#         image = cv2.resize(image, (64,64))
#         label = cv2.imread(label_list[a])[:,:,[2,1,0]]
#         label = cv2.resize(label, (64,64))
        
#         chose = random.randint(1,2)
#         #print(a)
#         if(chose == 1):
#             new_image = rotate(image)
#             new_label = rotate(label)
#         elif(chose == 4):
#             new_image = flip(image)
#             new_label = flip(label)
#         tmp1a[i]=new_image
#         tmp2a[i]=new_label
#     return tmp1a,tmp2a,A
     

def augment(image_list,label_list):
    fix=len(image_list)-1        #voglio lavorare solo sulle immagini della lista iniziale
    A = random.randint(10,fix)
    tmp1a = np.empty((A, 64, 64, 3), dtype=np.uint8)  #Qui ho N immagini
    tmp2a = np.empty((A, 64, 64, 3), dtype=np.uint8) 
    for i in range (0,A):
        a = random.randint(0,fix)
        image = cv2.imread(image_list[a])[:,:,[2,1,0]]
        image = cv2.resize(image, (64,64))
        label = cv2.imread(label_list[a])[:,:,[2,1,0]]
        label = cv2.resize(label, (64,64))
        
        chose = random.randint(1,5)
        #print(a)
        if(chose == 1):
            new_image = rotate(image)
            new_label = rotate(label)
        elif(chose == 2):
            new_image = brightness(image)
            new_label = label
        elif(chose == 3):
            new_image = saturate(image)
            new_label = label
        elif(chose == 4):
            new_image = flip(image)
            new_label = flip(label)
        elif(chose == 5):
            new_image = grayscale(image)
            new_label = label
        tmp1a[i]=new_image
        tmp2a[i]=new_label
    return tmp1a,tmp2a,A