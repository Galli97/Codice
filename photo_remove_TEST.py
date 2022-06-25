#####QUI TOLGO LE FOTO CHE NON HANNO UNA LABEL NELLA CARTELLA CORRISPONDENTE (LO ESEGUO UNA SOLA VOLTA)#######
import os
import cv2 
import numpy as np
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-images"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Test-labels"

path2 = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
path3 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"

dir = os.listdir(path2)
dir1 = os.listdir(path3)

print(len(dir));
print(len(dir1));

### CONSIDERO OGNI FOTO NELLA CARTELLA DELLE IMMAGINI
for i in range(15000,len(dir)):
    # new_dir = dir[i]       ###SOSTITUISCO IL TERMINE JPG CON png POICHè LE LABEL HANNO png NELLA STRINGA
    # new_dir1 = dir1[i]
    patt = str(dir[i])
    patt1 = str(dir1[i])

    image_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
    image = cv2.imread(image_path)
    label_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
    label = cv2.imread(label_path)

       
    new_image_path = os.path.join(path,patt)
    new_label_path = os.path.join(path1,patt1)
    new_image=cv2.imwrite(new_image_path, image)
    new_label=cv2.imwrite(new_label_path, label)

    delete_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
    delete_path1 = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels',patt1)
    os.remove(delete_path)
    os.remove(delete_path1)

#####ALLA FINE OTTENGO DUE CARTELLE CON 16064 FOTO########
     

