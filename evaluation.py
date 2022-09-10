import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from rete import *
from tensorflow.keras.optimizers import SGD
from utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.preprocessing.image import ImageDataGenerator

###### PERCORSO NEL DRIVE PER LAVORARE SU COLAB #########
# path = r"/content/drive/MyDrive/Tesi/image_patches_TEST.npy"
# path1 = r"/content/drive/MyDrive/Tesi/label_patches_TEST.npy"

# ####### PERCORSO IN LOCALE #########
# path = r"C:\Users\Mattia\Desktop\Tentativi128_2\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128_2\label_patches_TEST.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi128\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi128\label_patches_TEST.npy"

# path = r"C:\Users\Mattia\Desktop\Tentativi\image_patches_TEST.npy"
# path1 =  r"C:\Users\Mattia\Desktop\Tentativi\label_patches_TEST.npy"
# path2= r"C:\Users\Mattia\Documenti\Github\Codice\cropped_images_TEST.npy"
# path3 =  r"C:\Users\Mattia\Documenti\Github\Codice\cropped_labels_TEST.npy"

path = r"C:\Users\Mattia\Desktop\image_patches_TEST.npy"
path1 =  r"C:\Users\Mattia\Desktop\label_patches_TEST.npy"

### RECUPERO LE DUE LISTE SALVATE #####
tmp1 = get_np_arrays(path)          #recupero tmp1 dal file 
#print(type(tmp1))
print(tmp1.shape)
print('0: ',tmp1[1])
#tmp1=tmp1[0:500]

tmp2 = get_np_arrays(path1)          #recupero tmp2 dal file
#print(type(tmp2))
print(tmp2.shape)
print(len(tmp2))
print('0: ',tmp2[8,:,:,0])
#tmp2=tmp2[0:500]
print('tmp2: ', tmp2.shape)
# crop_images_list = get_np_arrays(path2)          #recupero tmp1 dal file 
# crop_labels_list = get_np_arrays(path3) 
SHAPE=128;
BATCH= 1
EPOCHS=10


# soil_count=0;
# bedrock_count=0;
# sand_count=0;
# bigrock_count=0;
# null_count=0;

# for i in range (0,len(tmp2)):
#     for r in range(0,SHAPE):
#         for c in range (0,SHAPE):
#             # if tmp1[i,r,c,:]!=0 and tmp2[i,r,c,:]!=2 and tmp2[i,r,c,:]!=3 and tmp2[i,r,c,:]!=4 and tmp2[i,r,c,:]!=0:
#             #     print(tmp1[i,r,c,:])
#             if tmp2[i,r,c,:]==4:
#                 soil_count+=1
#             elif tmp2[i,r,c,:]==1:
#                 bedrock_count+=1
#             elif tmp2[i,r,c,:]==2:
#                 sand_count+=1
#             elif tmp2[i,r,c,:]==3:
#                 bigrock_count+=1
#             elif tmp2[i,r,c,:]==0:
#                 null_count+=1
#             else:
#                 print(i)


#x_test = datagenerator(tmp1,tmp2,BATCH)
x_test = tf.data.Dataset.from_tensor_slices((tmp1, tmp2))
x_test = (
    x_test
    .batch(BATCH)
)

model = tf.keras.models.load_model('model.h5',custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU })

print("[INFO] Starting Evaluation")

predictions = model.predict(x_test,verbose=1,steps=len(tmp2))
print(predictions[5])
#print(np.argmax(predictions[0,0,0,:]))
# predictions = np.squeeze(predictions)
# predictions = np.argmax(predictions, axis=2)
print(predictions.shape)
save_predictions(predictions)

#true = decode_masks(tmp2)
prediction = decode_predictions(predictions,SHAPE)
cm1=np.ravel(tmp2)
print(cm1.shape)
cm2=np.ravel(prediction)
print(cm2.shape)
print(prediction[5].shape)
print(prediction[5])
matrix = tf.math.confusion_matrix(cm1,cm2,num_classes=5)
print(matrix)
matrix_nonull=matrix[1:4,1:4]
#128 count the # of pixels
# null_pixels =  null_count 
# bedrock_pixels= bedrock_count 
# sand_pixels= sand_count 
# bigrock_pixels= bigrock_count 
# soil_pixels= soil_count 

#128 1500 delle merged
null_pixels =  220661911
bedrock_pixels= 26192939
sand_pixels= 30639905
bigrock_pixels= 77698
soil_pixels= 60069019

#128 1500
# null_pixels =  9663358
# bedrock_pixels= 4869205
# sand_pixels= 3679903
# bigrock_pixels= 348196
# soil_pixels= 6015338

#128 500
# null_pixels = 3315038 
# bedrock_pixels =  1628395
# sand_pixels =  1241556
# bigrock_pixels =  66467
# soil_pixels =  1940544

#64
# null_pixels = 831340
# bedrock_pixels =  360564
# sand_pixels = 316722
# bigrock_pixels = 16502
# soil_pixels =  522872


matrix2 = np.array([[matrix[0]*100/null_pixels],[matrix[1]*100/bedrock_pixels], [matrix[2]*100/sand_pixels],[matrix[3]*100/bigrock_pixels], [matrix[4]*100/soil_pixels]])
np.set_printoptions(suppress=True)
print(matrix2.astype(float))

I = np.diag(matrix_nonull)
U = np.sum(matrix_nonull, axis=0) + np.sum(matrix_nonull, axis=1) - I
IOU = I/U
meanIOU = np.mean(IOU)

print(meanIOU)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# matconf = matrix2
# cmd_obj = ConfusionMatrixDisplay(matconf, display_labels=['null', 'bedrock', 'sand','bigrock','soil'])
# cmd_obj.plot()
# cmd_obj.ax_.set(
#                 title='Sklearn Confusion Matrix with labels!!', 
#                 xlabel='Predicted', 
#                 ylabel='Actual')
# plt.show()
matconf = confusion_matrix(cm1, cm2)
matconf = matconf.astype('float')*100.00 / matconf.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
cmd_obj = ConfusionMatrixDisplay(matconf, display_labels=['null', 'bedrock', 'sand','bigrock','soil'])
cmd_obj.plot(values_format=".1f")
cmd_obj.ax_.set(
                title='Confusion Matrix', 
                xlabel='Predicted', 
                ylabel='Actual')
plt.show()

model.evaluate(x_test,steps=len(tmp2))