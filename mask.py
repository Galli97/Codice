##Qui applico una maschera a tutte le labels per rendere più visibile la differenza tra le varie porzioni
import os 
import cv2 
import numpy as np
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Mask"

dir = os.listdir(path)
dir1 = os.listdir(path1)



####Considero un' immagine

bedrock=[1,1,1];
sand=[2,2,2];
bigrock=[3,3,3];

for x in range(len(dir1),len(dir)):
    img=str(dir[x])
    image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',img)
    image = cv2.imread(image_path)
    for i in range(0,1023):
        for j in range(0,1023): 
            channels_xy = image[i,j];          #SOIL is kept black, NULL (no label) is white 
            if all(channels_xy==bedrock):      #BEDROCK
                image[i,j,0]=255
                image[i,j,1]=0
                image[i,j,2]=0
            elif all(channels_xy==sand):    #SAND
                image[i,j,0]=0
                image[i,j,1]=255
                image[i,j,2]=0
            elif all(channels_xy==bigrock):    #BIG ROCK
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=255
    new_img=str(dir[x])
    new_image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Mask',new_img)
    new_image=cv2.imwrite(new_image_path, image)

#print(np.shape(image))  #(1024,1024,3)  
#cv2.imshow('immagine',new_image)
#####waits for user to press any key (this is necessary to avoid Python kernel form crashing)
#cv2.waitKey(0) 
######closing all open windows when the user press a key
#cv2.destroyAllWindows() 