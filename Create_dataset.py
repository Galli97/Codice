#####CREO UN DATASET IL PIù POSSIBILE BILANCIATO#######
import os
import shutil
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"


newpath = r'C:\Users\Mattia\Desktop\Train_images'
newpath1 = r'C:\Users\Mattia\Desktop\Train_labels'


dir = os.listdir(path)
dir1 = os.listdir(path1)


###QUESTO CICLO FOR SELEZIONA LE IMMAGINI E LE LABEL DALLE RISPETTIVE CARTELLE, PRENDENDONE UNA OGNI 10, PER UN TOTALE DI 1000 IMMAGINI E 1000 LABEL####
j=0
for i in range(0,len(dir)):
    if(j==1000):
        break
    if(i%10==0):
        patt=str(dir[i])
        patt1=str(dir1[i])
        path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
        path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
        shutil.copy(path, newpath)
        shutil.copy(path1, newpath1)
        j+=1

#####TROVO LE IMMAGINI E LE LABEL NELL CARTELLE Train_images, Train_labels########