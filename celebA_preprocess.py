import glob
from os import sep, getcwd,path
import numpy as np
from PIL import Image

#def crop(im,out_size=64):
#    imresize(im,(out_size,out_size))

if __name__=='__main__':
    out_size=64,64
    dataset_path=getcwd()+sep+'dataset'+sep+'celebA'+sep
    img_list=glob.glob(dataset_path+'*.jpg')
    print(path.join(getcwd(),"dataset","celebA_64"))
    print('total image count of '+str(len(img_list)))
    count=0
    for img_dic in img_list:
        _, file = path.split(img_dic)
        img=Image.open(img_dic)
        img_crop=img.crop((10,30,168,188))
        img_crop.thumbnail(out_size)
        img_crop.save(path.join(getcwd(),"dataset","celebA_64",file),"JPEG")
        count=count+1
        if count%1000==0:
            print('now at '+str(count))