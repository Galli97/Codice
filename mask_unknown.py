import os 
import cv2 
import numpy as np

### CREO DUE LISTE CON I PATH DELLE IMMAGINI E DELLE MASK 
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Mask"
path2 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Mask_unknown"

dir = os.listdir(path)
dir1 = os.listdir(path1)   
dir2= os.listdir(path2) ##QUESTO LISTA MI SERVE PER RIPRENDERE L'OPERAZIONE DA DOVE AVEVO INTERROTTO

uk128=[128,128,128];
uk64=[64,64,64];
img=str(dir[340])
image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',img)
image = cv2.imread(image_path)
print(image)
for x in range(len(dir2),len(dir1)):
    img=str(dir[x])
    image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',img)
    image = cv2.imread(image_path)
    for i in range(0,1024):
        for j in range(0,1024): 
            channels_xy = image[i,j];         
            if all(channels_xy==uk128):      #[128,128,128]
                print(x)
                image[i,j,0]=255
                image[i,j,1]=0
                image[i,j,2]=0
            elif all(channels_xy==uk64):    #[64,64,64]
                print(x)
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=255
    new_img=str(dir[x])
    new_image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Mask_unknown',new_img)
    new_image=cv2.imwrite(new_image_path, image)