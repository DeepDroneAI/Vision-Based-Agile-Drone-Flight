#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################Import-Libraries####################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F

from dataprocessing import DatasetProcessing
from utils import weighted_mse_loss
from collections import OrderedDict
from Dronet import ResNet, BasicBlock


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------------------------

#-------------------------------Data Path------------------------------------------

#ata_path = '/home/deepdrone/Dataset/OurBasic'

data_path = "/home/recep/dataprocessing/data"

train_data = 'train'

train_label_file = 'train_labels.txt'

train_image_file = 'train_img.txt'

val_data = 'val'

val_label_file = 'val_labels.txt'

val_image_file = 'val_img.txt'

#print("###############DataPath is given.#################")
#----------------------------------------------------------------------------------

#--------------------------------Import-Dataset------------------------------------
transformations = transforms.Compose([
        transforms.Resize([200, 200]),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.ToTensor()]
        )

#transforms.CenterCrop(200),
#transforms.ColorJitter(hue=.05, saturation=.05),
#transforms.Grayscale(num_output_channels=1),
        
dset_train = DatasetProcessing(
    data_path, train_data, train_image_file, train_label_file, transformations)

dset_val = DatasetProcessing(
    data_path, val_data, val_image_file, val_label_file, transformations)


image_datasets = {}
image_datasets["train"] = dset_train
image_datasets["val"] = dset_val

dataset_sizes = {}
dataset_sizes["train"] = len(image_datasets["train"])
dataset_sizes["val"] = len(image_datasets["val"])
#print("Traning %d images, Validation on %d images" % (
#    len(image_datasets["train"]), (len(image_datasets["val"])) ))

#----------------------------------------------------------------------------------


#---------------------------Train Model--------------------------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_loss = 1
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)                
                                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train       
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)            
 
                    # Loss implementation
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()                        
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                print("Running Loss: ",running_loss,"Loss: ",loss)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_losses[phase].append(epoch_loss)

                 
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(),'/home/recep/dataprocessing/model/{}_{}_{}_loss_{:.4f}_PG.pth'.format(batch_size,lr,epoch, epoch_loss))
                print("{}_{}_{}_Pg.pth file is saved".format(batch_size,lr,epoch))
                  
                  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model
#----------------------------------------------------------------------------------

#---------------------------------DataLoader---------------------------------------
# #Initilized our parameters
if __name__ == "__main__":
    batch_size_list = [16]
    lr_list = [0.001]
    print(device)
    print (os.getcwd())
    for batch_size in batch_size_list:
        for lr in lr_list:
           
            print("Lists are uploading...")
            num_workers = 4

            epoch_losses = {"train":[], "val":[]}

            print("/////////batch_size/////////:", batch_size, "/////////learning_rate/////////",lr)

            train_loader = DataLoader(dset_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers
                                )

            val_loader = DataLoader(dset_val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers
                                )
            print("DataLoader is loaded with {} number of worker and {} batch size.".format(num_workers, batch_size))

            dataloaders = {}
            dataloaders["train"] = train_loader
            dataloaders["val"] = val_loader
            #----------------------------------------------------------------------------------


            #--------------------------------------Criterion-----------------------------------
            #criterion = weighted_mse_loss(inputs, labels, 1, 3, 20) 
            criterion = nn.MSELoss()
            #----------------------------------------------------------------------------------


            #---------------------------------Import Model-------------------------------------
            model_ft =  ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
            model_ft.load_state_dict(torch.load('/home/recep/deep-drone-traj/weights/Dronet_new.pth'))   
            #print(model_ft)
            print("Model is Updated.")
            model_ft = model_ft.to(device)
            #----------------------------------------------------------------------------------

            #-----------------------------------Optimizer--------------------------------------
            #optimizer_ft = optim.SGD(model_ft.parameters(), lr = lr, momentum=0.9)
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay = 0.001)
            epochh = 50
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=0.001, gamma=0.1) #step size = lr
            #----------------------------------------------------------------------------------
            
            #--------------------------------------CallForTrain--------------------------------
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs = epochh)
            #----------------------------------------------------------------------------------


            #---------------------------------PLOT---------------------------------------------
            plt.figure()

            plt.plot(epoch_losses['train'], label = "Train Loss")
            plt.plot(epoch_losses['val'], label = "Val Loss")
            plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=.5)

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid('on')
            plt.title('Train')

            plt.show()
            plt.savefig("LowCorelasyon_{}_{}.png".format(batch_size,lr))
            #print(model_ft)
            #----------------------------------------------------------------------------------