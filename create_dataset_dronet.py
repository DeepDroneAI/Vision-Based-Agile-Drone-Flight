import os 
import numpy as np
from shutil import copyfile

folder_index=170
max_index=180
image_index=26683
folder_name="/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/images/images-"
dest_folder="/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/images/val_data/"

image_name_arr=[]
gts_data=[]


for i in range(170,max_index):
    folder_index=i
    folder_name_current=folder_name+str(folder_index)+"/"
    image_labes=folder_name_current+"image_labels-"+str(folder_index)+".txt"
    image_names=folder_name_current+"image_names-"+str(folder_index)+".txt"

    label_data=open(image_labes,'r')
    Lines=label_data.readlines()
    for line in Lines:
        gts_data.append(line)
    name_data=open(image_names,'r')
    Lines=name_data.readlines()
    for line in Lines:
        image_name_arr.append(line)

    while True:
        image_name="frame000"+str(image_index)+".png"
        image_statu=os.path.exists(folder_name_current+image_name)
        if image_statu==True:
            copyfile(folder_name_current+image_name,dest_folder+image_name)
            image_index+=1
        else:
            break


with open("/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/images/val_labels.txt",'w') as f:
    for item in gts_data:
        f.write("%s"%item)
with open("/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/images/val_im_names.txt",'w') as f:
    for item in image_name_arr:
        f.write("%s"%item)