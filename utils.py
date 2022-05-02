import numpy as np
import tensorflow as tf
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


def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label