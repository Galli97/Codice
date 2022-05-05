import numpy as np
import tensorflow as tf
import random
import cv2

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
        
        chose = random.randint(1,2)
        #print(a)
        if(chose == 1):
            new_image = rotate(image)
            new_label = rotate(label)
        elif(chose == 4):
            new_image = flip(image)
            new_label = flip(label)
        tmp1a[i]=new_image
        tmp2a[i]=new_label
    return tmp1a,tmp2a,A
     

# def augment(image_list,label_list):
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
        
#         chose = random.randint(1,5)
#         #print(a)
#         if(chose == 1):
#             new_image = rotate(image)
#             new_label = rotate(label)
#         elif(chose == 2):
#             new_image = brightness(image)
#             new_label = brightness(label)
#         elif(chose == 3):
#             new_image = saturate(image)
#             new_label = saturate(label)
#         elif(chose == 4):
#             new_image = flip(image)
#             new_label = flip(label)
#         elif(chose == 5):
#             new_image = grayscale(image)
#             new_label = grayscale(label)
#         tmp1a[i]=new_image
#         tmp2a[i]=new_label
#     return tmp1a,tmp2a,A