#####CREO UN DATASET IL PIù POSSIBILE BILANCIATO#######
import os
import shutil
from matplotlib import image
import cv2
import numpy as np

init_path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
init_path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"


newpath = r'C:\Users\Mattia\Desktop\Train_images'
newpath1 = r'C:\Users\Mattia\Desktop\Train_labels'


dir = os.listdir(init_path)
dir1 = os.listdir(init_path1)


#### DEFINISCO GLI ARRAY DEI VARI PIXEL

soil=[0,0,0];
bedrock=[1,1,1];
sand=[2,2,2];
bigrock=[3,3,3];

flag_sand=False;
flag_bedrock=False;
flag_bigrock=False;
flag_soil=False;
count=0;

### PRENDO UN'IMMAGINE E CAMBIO I PIXEL CORRISPONDENTI ALLE CLASSI BEDROCK SAND E BIGROCK ATTRIBUENDO AD OGNUNO UN COLORE DIVERSO
#da 600-1587 prendo immagini che abbiano o la big rock, oppure sand-soil-bedrock
#Da 3000-3554  prendo immagini che abbiano o la big rock, oppure sand-soil
#Da 3554-4168  prendo immagini che abbiano o la big rock, oppure sand
#Da 4168 prendo immagini che abbiano o la big rock, oppure almeno altre due classi
#Da 7000-7140 prendo immagini che abbiano o la big rock, oppure almeno altre due classi
#7140-8447
#8447-9455
for x in range(8447,len(dir1)):
    print('Label: ', x)
    flag_sand=False;
    flag_bedrock=False;
    flag_bigrock=False;
    flag_soil=False;
    if(count==500):
        break
    else:
        img=str(dir1[x])
        #print(img)
        image_path = os.path.join(init_path1,img)
        image = cv2.imread(image_path)
        #print(image.shape)
        for i in range(0,1024):
            if(flag_bigrock==True):
                break
            elif(flag_sand==True and flag_soil==True):
                break
            elif(flag_sand==True and flag_bedrock==True):
                break
            elif(flag_bedrock==True and flag_soil==True):
                break
            for j in range(0,1024): 
                channels_xy = image[i,j];
                if(flag_bigrock==True):
                    break
                elif(flag_sand==True and flag_soil==True):
                    break
                elif(flag_sand==True and flag_bedrock==True):
                    break
                elif(flag_bedrock==True and flag_soil==True):
                    break
                elif all(channels_xy==bedrock):      #BEDROCK
                    flag_bedrock=True
                elif all(channels_xy==sand):    #SAND
                    flag_sand=True
                elif all(channels_xy==bigrock):    #BIG ROCK
                    flag_bigrock=True
                elif all(channels_xy==soil):    #BIG ROCK
                    flag_soil=True
        if (flag_bigrock==True):
            print('Big Rock IN')
            patt=str(dir[x])
            patt1=str(dir1[x])
            path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
            path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
            shutil.copy(path, newpath)
            shutil.copy(path1, newpath1)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print(count)
        elif (flag_sand==True and flag_soil==True):
            print('Sand-soil')
            patt=str(dir[x])
            patt1=str(dir1[x])
            path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
            path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
            shutil.copy(path, newpath)
            shutil.copy(path1, newpath1)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print(count)
        elif (flag_sand==True and flag_bedrock==True):
            print('Sand-bedrock')
            patt=str(dir[x])
            patt1=str(dir1[x])
            path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
            path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
            shutil.copy(path, newpath)
            shutil.copy(path1, newpath1)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print(count)
        elif (flag_bedrock==True and flag_soil==True):
            print('Soil-Bedrock')
            patt=str(dir[x])
            patt1=str(dir1[x])
            path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
            path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
            shutil.copy(path, newpath)
            shutil.copy(path1, newpath1)
            flag_sand=False;
            flag_bedrock=False;
            flag_bigrock=False;
            flag_soil=False;
            count+=1
            print(count)

###QUESTO CICLO FOR SELEZIONA LE IMMAGINI E LE LABEL DALLE RISPETTIVE CARTELLE, PRENDENDONE UNA OGNI 10, PER UN TOTALE DI 1000 IMMAGINI E 1000 LABEL####
# j=0
# for i in range(0,len(dir)):
#     if(j==1000):
#         break
#     if(i%10==0):
#         patt=str(dir[i])
#         patt1=str(dir1[i])
#         path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
#         path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
#         shutil.copy(path, newpath)
#         shutil.copy(path1, newpath1)
#         j+=1

#####TROVO LE IMMAGINI E LE LABEL NELL CARTELLE Train_images, Train_labels########