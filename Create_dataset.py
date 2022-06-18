#####CREO UN DATASET IL PIù POSSIBILE BILANCIATO#######
import os
import shutil
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"


newpath = r'C:\Users\Mattia\Desktop\Train_images'
newpath1 = r'C:\Users\Mattia\Desktop\Train_labels'


dir = os.listdir(path)
dir1 = os.listdir(path1)

# print(len(dir));
# print(len(dir1));
j=0
for i in range(0,len(dir)):
    if(j==400):
        break
    if(i%100):
        patt=str(dir[i])
        patt1=str(dir1[i])
        newpath = os.path.join(r'C:\Users\Mattia\Desktop\Train_images',patt)
        newpath1 = os.path.join(r'C:\Users\Mattia\Desktop\Train_labels',patt1)
        shutil.copyfile(path, newpath)
        shutil.copyfile(path1, newpath1)
        j+=1

#####PRENDO 400 IMMAGINI########