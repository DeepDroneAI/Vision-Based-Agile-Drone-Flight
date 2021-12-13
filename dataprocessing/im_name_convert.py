import os


im_name="/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/dataprocessing/Dataset_Dronet/train_im_names.txt"

f=open(im_name,'r')
Lines=f.readlines()

image_name_arr=[]

for line in Lines:
    image_name=line.split("/")[3]
    image_name_arr.append(image_name)

with open("/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/dataprocessing/Dataset_Dronet/train_im_names2.txt",'w') as f:
    for item in image_name_arr:
        f.write("%s"%item)


im_name="/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/dataprocessing/Dataset_Dronet/val_im_names.txt"

f=open(im_name,'r')
Lines=f.readlines()

image_name_arr=[]

for line in Lines:
    image_name=line.split("/")[3]
    image_name_arr.append(image_name)

with open("/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight/dataprocessing/Dataset_Dronet/val_im_names2.txt",'w') as f:
    for item in image_name_arr:
        f.write("%s"%item)