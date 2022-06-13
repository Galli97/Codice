import numpy as np
import tensorflow as tf
import random
import cv2
import tensorflow as tf




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

##########################
########################################################
#######################################################

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

    processed_image = image_list
    label =label_list
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

