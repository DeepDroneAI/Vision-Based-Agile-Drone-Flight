# Extras for Perception
from time import process_time
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image
import cv2

import Dronet
import lstmf
import numpy as np
import random


class Single_Dronet:
    def __init__(self):
        self.base_path="/home/drone-ai/Documents/Traj_Test"
        self.model_path="/weights/16_0.001_2_loss_0.0101_PG.pth"
        self.image_path="/Dronet_Train_Data/test/"
        # Dronet
        self.device = torch.device('cpu')
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load(self.base_path+self.model_path,map_location=self.device))   
        self.Dronet.eval()

        self.index=0
        self.alpha=random.uniform(-1.5,1)

        self.label_list=[]
        self.imae_name_list=[]

        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.period_denum = 30.

        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            #transforms.Lambda(self.gaussian_blur),
                            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                            transforms.ToTensor()])

    def run_dronet(self,img_rgb):
        img =  Image.fromarray(img_rgb)
        image = self.transformation(img)
        pose_gate_body = self.Dronet(image)
        pose_gate_body = pose_gate_body.detach().numpy().reshape(-1,1)
        return pose_gate_body  

    def increase_brightness(self,image, alpha=1):
        new_image = np.zeros(image.shape, image.dtype)
        alpha = alpha # Simple contrast control
        beta = 0    # Simple brightness control
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
        return new_image



    def get_image(self,index):
        image_name="frame000"+str(index)+".png"
        image=cv2.imread(self.base_path+self.image_path+image_name)

        return image

    def calculate_loss(self,data1,data2):
        loss=0
        for i in range(4):
            loss+=100*(5 if i==1 or i==2 else 1)*pow(data1[i][0]-data2[i][0],2)

        return loss


    def run(self):
        image=self.get_image(self.index)
        pose_gate_body_1=self.run_dronet(image)
        image=self.increase_brightness(image,alpha=10**random.uniform(-1.5,1))
        image_name="frame000"+str(self.index)+".png"
        self.imae_name_list.append(image_name)
        cv2.imwrite("CovNet_Data/test/"+image_name,image)
        pose_gate_body=self.run_dronet(image)
        loss=self.calculate_loss(pose_gate_body,pose_gate_body_1)
        self.label_list.append(loss)
        return pose_gate_body

    def data_generate(self):
        for i in range(2210,2483):
            print("Index:{}".format(i))
            self.index=i
            self.run()
        labels=np.array(self.label_list).reshape((len(self.label_list),1))
        np.savetxt("CovNet_Data/test_labels.txt",labels)
        names=np.array(self.imae_name_list).reshape((len(self.imae_name_list),1))
        np.savetxt("CovNet_Data/test_img.txt",names, delimiter=" ", fmt="%s")


dronet=Single_Dronet()
dronet.data_generate()


        

    