#####QUI TOLGO LE FOTO CHE NON HANNO UNA LABEL NELLA CARTELLA CORRISPONDENTE (LO ESEGUO UNA SOLA VOLTA)#######
import os
path = r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images"
path1 =  r"C:\Users\Mattia\Desktop\Tesi\Dataset\Train-labels"

dir = os.listdir(path)
dir1 = os.listdir(path1)

print(len(dir));
print(len(dir1));

for i in range(0,len(dir)):
    a=0;
    new_dir=dir[i].replace("JPG", "png");
    #print(new_dir)
    for j in range(0,len(dir1)):
        if(new_dir==dir1[j]):
            a+=1
    if(a==0):
     #print(i);
     patt=str(dir[i])
     delete_path = os.path.join(r'C:\Users\Mattia\Desktop\Tesi\Dataset\Train-images',patt)
     #print(delete_path)
     os.remove(delete_path)

#####ALLA FINE OTTENGO DUE CARTELLE CON 16064 FOTO########
     

