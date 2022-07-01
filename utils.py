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
    A = random.randint(2000,fix)
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
        chose = random.randint(1,3)
        #print(a)
        if(chose == 1):
            new_image = rotate(image)
            new_label = rotate(label)
        elif(chose == 2):
            new_image = brightness(image)
            new_label = label
        elif(chose == 3):
            new_image = flip(image)
            new_label = flip(label)
        elif(chose == 4):
            new_image = grayscale(image)
            new_label = label
        elif(chose == 5):
            new_image = saturate(image)
            new_label = label
        elif(chose == 6):
            new_image = contrast(image)
            new_label = label
        # elif(chose == 4):
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

def _adjust_labels(labels, predictions):
  """Adjust the 'labels' tensor by squeezing it if needed."""
  labels = tf.cast(labels, tf.int32)
  if len(predictions.shape) == len(labels.shape):
    labels = tf.squeeze(labels, [-1])
  return labels, predictions


def _validate_rank(labels, predictions, weights):
  if weights is not None and len(weights.shape) != len(labels.shape):
    raise RuntimeError(
        ("Weight and label tensors were not of the same rank. weights.shape "
         "was %s, and labels.shape was %s.") %
        (predictions.shape, labels.shape))
  if (len(predictions.shape) - 1) != len(labels.shape):
    raise RuntimeError(
        ("Weighted sparse categorical crossentropy expects `labels` to have a "
         "rank of one less than `predictions`. labels.shape was %s, and "
         "predictions.shape was %s.") % (labels.shape, predictions.shape))


def loss(labels, predictions, weights=None, from_logits=False):
  """Calculate a per-batch sparse categorical crossentropy loss.
  This loss function assumes that the predictions are post-softmax.
  Args:
    labels: The labels to evaluate against. Should be a set of integer indices
      ranging from 0 to (vocab_size-1).
    predictions: The network predictions. Should have softmax already applied.
    weights: An optional weight array of the same shape as the 'labels' array.
      If None, all examples will be used.
    from_logits: Whether the input predictions are logits.
  Returns:
    A loss scalar.
  Raises:
    RuntimeError if the passed tensors do not have the same rank.
  """
  # When using these functions with the Keras core API, we will need to squeeze
  # the labels tensor - Keras adds a spurious inner dimension.
  labels, predictions = _adjust_labels(labels, predictions)
  _validate_rank(labels, predictions, weights)

  example_losses = tf.keras.losses.sparse_categorical_crossentropy(
      labels, predictions, from_logits=from_logits)

  if weights is None:
    return tf.reduce_mean(example_losses)
  weights = tf.cast(weights, predictions.dtype)
  return tf.math.divide_no_nan(
      tf.reduce_sum(example_losses * weights), tf.reduce_sum(weights))



def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def weighted_cross_entropy(beta):
  def loss(y_true, y_pred):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = 1 - tf.cast(y_true, tf.float32)
    
    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
    return tf.reduce_mean(o)

  return loss


def iou_coef(y_true, y_pred, smooth=1):
  y_true=tf.cast(y_true, tf.float32)
  y_pred=tf.cast(y_pred, tf.float32)
  # print('ypre: ',y_pred.shape)
  # print('ytrue: ',y_true.shape)
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
##Calculate Intersection Over Union Score for predicted layer
import numpy as np
import scipy.misc as misc

def GetIOU(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union

    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassIOU[i]))
       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight
def IOU(y_pred,ytrue,num_classes,Classes):
   CIOU,CU=GetIOU(y_pred,y_true,num_classes,Classes) #Calculate intersection over union
   Intersection+=CIOU*CU
   Union+=CU
   for i in range(len(Classes)):
        if Union[i]>0: print(Classes[i]+"\t"+str(Intersection[i]/Union[i]))

# class MyMeanIOU(tf.keras.metrics.MeanIoU):
#     def update_state(self, y_true, y_pred, sample_weight=[0.1,0.3,0.4,0.2,0]):
#         return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

class MyMeanIoU(tf.keras.metrics.MeanIoU):
    '''Custom meanIoU to handle borders (class = -1).'''
    def update_state(self, y_true, y_pred_onehot, sample_weight=None):
        y_pred = tf.argmax(y_pred_onehot, axis=-1)
        ## add 1 so boundary class=0
        y_true = tf.cast(y_true+1, self._dtype)
        y_pred = tf.cast(y_pred+1, self._dtype)
        ## Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        ## calculate confusion matrix with one extra class
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes+1,
            weights=sample_weight,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm[1:, 1:]) # remove boundary

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
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tf.constant([1.0, 3.0, 9.0, 8.0, 4.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights
########################

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
            resize_to_range(
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
            scale = get_random_scale(
                min_scale_factor, max_scale_factor, scale_factor_step_size)
            processed_image, label = randomly_scale_image_and_label(
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
            processed_image, label = random_crop(
                [processed_image, label], crop_height, crop_width)

        processed_image.set_shape([crop_height, crop_width, 3])

        if label is not None:
            label.set_shape([crop_height, crop_width, 1])

        if is_training:
            # Randomly left-right flip the image and label.
            processed_image, label, _ = flip_dim(
                [processed_image, label], _PROB_OF_FLIP, dim=1)

        return original_image, processed_image, label

        
def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=tf.image.ResizeMethod.BILINEAR):
  """Resizes image or label so their sides are within the provided range.
  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum size is equal to min_size
     without the other side exceeding max_size, then do so.
  2. Otherwise, resize so the largest side is equal to max_size.
  An integer in `range(factor)` is added to the computed sides so that the
  final dimensions are multiples of `factor` plus one.
  Args:
    image: A 3D tensor of shape [height, width, channels].
    label: (optional) A 3D tensor of shape [height, width, channels] (default)
      or [channels, height, width] when label_layout_is_chw = True.
    min_size: (scalar) desired size of the smaller image side.
    max_size: (scalar) maximum allowed size of the larger image side. Note
      that the output dimension is no larger than max_size and may be slightly
      smaller than min_size when factor is not None.
    factor: Make output size multiple of factor plus one.
    align_corners: If True, exactly align all 4 corners of input and output.
    label_layout_is_chw: If true, the label has shape [channel, height, width].
      We support this case because for some instance segmentation dataset, the
      instance segmentation is saved as [num_instances, height, width].
    scope: Optional name scope.
    method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.
  Returns:
    A 3-D tensor of shape [new_height, new_width, channels], where the image
    has been resized (with the specified method) so that
    min(new_height, new_width) == ceil(min_size) or
    max(new_height, new_width) == ceil(max_size).
  Raises:
    ValueError: If the image is not a 3D tensor.
  """
  with tf.name_scope(scope, 'resize_to_range', [image]):
    new_tensor_list = []
    min_size = tf.to_float(min_size)
    if max_size is not None:
      max_size = tf.to_float(max_size)
      # Modify the max_size to be a multiple of factor plus 1 and make sure the
      # max dimension after resizing is no larger than max_size.
      if factor is not None:
        max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                    - factor)

    [orig_height, orig_width, _] = resolve_shape(image, rank=3)
    orig_height = tf.to_float(orig_height)
    orig_width = tf.to_float(orig_width)
    orig_min_size = tf.minimum(orig_height, orig_width)

    # Calculate the larger of the possible sizes
    large_scale_factor = min_size / orig_min_size
    large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
    large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
    large_size = tf.stack([large_height, large_width])

    new_size = large_size
    if max_size is not None:
      # Calculate the smaller of the possible sizes, use that if the larger
      # is too big.
      orig_max_size = tf.maximum(orig_height, orig_width)
      small_scale_factor = max_size / orig_max_size
      small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
      small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
      small_size = tf.stack([small_height, small_width])
      new_size = tf.cond(
          tf.to_float(tf.reduce_max(large_size)) > max_size,
          lambda: small_size,
          lambda: large_size)
    # Ensure that both output sides are multiples of factor plus one.
    if factor is not None:
      new_size += (factor - (new_size - 1) % factor) % factor
    new_tensor_list.append(tf.image.resize_images(
        image, new_size, method=method, align_corners=align_corners))
    if label is not None:
      if label_layout_is_chw:
        # Input label has shape [channel, height, width].
        resized_label = tf.expand_dims(label, 3)
        resized_label = tf.image.resize_nearest_neighbor(
            resized_label, new_size, align_corners=align_corners)
        resized_label = tf.squeeze(resized_label, 3)
      else:
        # Input label has shape [height, width, channel].
        resized_label = tf.image.resize_images(
            label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            align_corners=align_corners)
      new_tensor_list.append(resized_label)
    else:
      new_tensor_list.append(None)
    return new_tensor_list

def get_random_scale(min_scale_factor, max_scale_factor, step_size):
  """Gets a random scale value.
  Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.
  Returns:
    A random scale value selected between minimum and maximum value.
  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
  if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
    raise ValueError('Unexpected value of min_scale_factor.')

  if min_scale_factor == max_scale_factor:
    return tf.to_float(min_scale_factor)

  # When step_size = 0, we sample the value uniformly from [min, max).
  if step_size == 0:
    return tf.random_uniform([1],
                             minval=min_scale_factor,
                             maxval=max_scale_factor)

  # When step_size != 0, we randomly select one discrete value from [min, max].
  num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
  scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
  shuffled_scale_factors = tf.random_shuffle(scale_factors)
  return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
  """Randomly scales image and label.
  Args:
    image: Image with shape [height, width, 3].
    label: Label with shape [height, width, 1].
    scale: The value to scale image and label.
  Returns:
    Scaled image and label.
  """
  # No random scaling if scale == 1.
  if scale == 1.0:
    return image, label
  image_shape = tf.shape(image)
  new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

  # Need squeeze and expand_dims because image interpolation takes
  # 4D tensors as input.
  image = tf.squeeze(tf.image.resize_bilinear(
      tf.expand_dims(image, 0),
      new_dim,
      align_corners=True), [0])
  if label is not None:
    label = tf.squeeze(tf.image.resize_nearest_neighbor(
        tf.expand_dims(label, 0),
        new_dim,
        align_corners=True), [0])

  return image, label

def random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]

def flip_dim(tensor_list, prob=0.5, dim=1):
  """Randomly flips a dimension of the given tensor.
  The decision to randomly flip the `Tensors` is made together. In other words,
  all or none of the images pass in are flipped.
  Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
  that we can control for the probability as well as ensure the same decision
  is applied across the images.
  Args:
    tensor_list: A list of `Tensors` with the same number of dimensions.
    prob: The probability of a left-right flip.
    dim: The dimension to flip, 0, 1, ..
  Returns:
    outputs: A list of the possibly flipped `Tensors` as well as an indicator
    `Tensor` at the end whose value is `True` if the inputs were flipped and
    `False` otherwise.
  Raises:
    ValueError: If dim is negative or greater than the dimension of a `Tensor`.
  """
  random_value = tf.random_uniform([])

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
  '''Resizes the images contained in a 4D tensor of shape
  - [batch, channels, height, width] (for 'channels_first' data_format)
  - [batch, height, width, channels] (for 'channels_last' data_format)
  by a factor of (height_factor, width_factor). Both factors should be
  positive integers.
  '''
  if data_format == 'default':
      data_format = K.image_data_format()
  if data_format == 'channels_first':
      original_shape = K.int_shape(X)
      if target_height and target_width:
          new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
      else:
          new_shape = tf.shape(X)[2:]
          new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
      X = permute_dimensions(X, [0, 2, 3, 1])
      X = tf.compat.v1.image.resize_bilinear(X, new_shape)
      X = permute_dimensions(X, [0, 3, 1, 2])
      if target_height and target_width:
          X.set_shape((None, None, target_height, target_width))
      else:
          X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
      return X
  elif data_format == 'channels_last':
      original_shape = K.int_shape(X)
      if target_height and target_width:
          new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
      else:
          new_shape = tf.shape(X)[1:3]
          new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
      X = tf.compat.v1.image.resize_bilinear(X, new_shape)
      if target_height and target_width:
          X.set_shape((None, target_height, target_width, None))
      else:
          X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
      return X
  else:
      raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(Layer):
  def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
      if data_format == 'default':
          data_format = K.image_data_format()
      self.size = tuple(size)
      if target_size is not None:
          self.target_size = tuple(target_size)
      else:
          self.target_size = None
      assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
      self.data_format = data_format
      self.input_spec = [InputSpec(ndim=4)]
      super(BilinearUpSampling2D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
      if self.data_format == 'channels_first':
          width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
          height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
          if self.target_size is not None:
              width = self.target_size[0]
              height = self.target_size[1]
          return (input_shape[0],
                  input_shape[1],
                  width,
                  height)
      elif self.data_format == 'channels_last':
          width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
          height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
          if self.target_size is not None:
              width = self.target_size[0]
              height = self.target_size[1]
          return (input_shape[0],
                  width,
                  height,
                  input_shape[3])
      else:
          raise Exception('Invalid data_format: ' + self.data_format)

  def call(self, x, mask=None):
      if self.target_size is not None:
          return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
      else:
          return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

  def get_config(self):
      config = {'size': self.size, 'target_size': self.target_size}
      base_config = super(BilinearUpSampling2D, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

