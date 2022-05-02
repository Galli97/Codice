import numpy as np
import tensorflow as tf
import random

def save_np_arrays(tmp1):
    with open('image_arrays.npy','wb') as f:
        np.save(f,tmp1)

def get_np_arrays(file):
    with open(file,'rb') as f:
        tmp1 = np.load(f)
    return tmp1

def save_np_arrays_labels(tmp2):
    with open('label_arrays.npy','wb') as f:
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

# def augment(image_list,tmp1,tmp2):
#     n = random.randint(10,100)
#     for i in range n:
#         a = random.randint(0,len(image_list)-1)
#         image = cv2.imread(image_list[a])[:,:,[2,1,0]]
#         new_image = cropp(image,0.5)
#         tmp1_augmented.append(new_image)
#         label = cv2.imread(label_list[a])[:,:,[2,1,0]]
#         new_label = cropp(label,0.5)
#         tmp2_augmented.append(new_label)
     