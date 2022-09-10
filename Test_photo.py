# #####QUI TOLGO LE FOTO CHE NON HANNO UNA LABEL NELLA CARTELLA CORRISPONDENTE (LO ESEGUO UNA SOLA VOLTA)#######
import os
path = r"C:\Users\Mattia\Desktop\TEST_images"
#path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\ai4mars-dataset-merged-0.1\ai4mars-dataset-merged-0.1\msl\labels\test\masked-gold-min3-100agree"

dir = os.listdir(path)
dir1 = os.listdir(path1)

print(len(dir));
print(len(dir1));

# ### CONSIDERO OGNI FOTO NELLA CARTELLA DELLE IMMAGINI
for i in range(0,len(dir)):
    a=0;
    new_dir=dir[i].replace(".JPG", "_merged.png");       ###SOSTITUISCO IL TERMINE JPG CON png POICHè LE LABEL HANNO png NELLA STRINGA
    for j in range(0,len(dir1)):                ###CONTROLLO CHE NELLA CARTELLA DELLE LABEL CI SIA UNA LABEL ASSOCIATA ALL'IMMAGINE CONSIDERATA
        if(new_dir==dir1[j]):
            a+=1
            print(a)
    if(a==0):                                   ### SE NON TROVO LA LABEL, RIMUOVO LA FOTO/FILE DALLA CARTELLA DELLE IMMAGINI
     #print(i);
     patt=str(dir[i])
     delete_path = os.path.join(r'C:\Users\Mattia\Desktop\TEST_images',patt)
     #print(delete_path)
     os.remove(delete_path)

dir = os.listdir(path)
print(len(dir));
# #####ALLA FINE OTTENGO DUE CARTELLE CON 16064 FOTO########
     



