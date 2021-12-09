#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------------
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
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F

from dataprocessing import DatasetProcessing
from utils import weighted_mse_loss
from collections import OrderedDict
from Dronet import ResNet, BasicBlock






#---------------------------Train Model--------------------------------------------
def train_model(model, num_epochs=25, batch_size=16, lr=0.0001, weight_decay = 0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #-------------------------------Data Path------------------------------------------
    print (os.getcwd())

    data_path = 'C:\\Users\deepd\OneDrive\Masaüstü\Dataset\Sequence'

    train_data = 'train'

    train_label_file = 'train_label.txt'

    train_image_file = 'train_img.txt'

    val_data = 'val'

    val_label_file = 'val_label.txt'

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
    #
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
    print("Traning %d images, Validation on %d images" % (
        len(image_datasets["train"]), (len(image_datasets["val"])) ))

    #----------------------------------------------------------------------------------
    train_loader = DataLoader(dset_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4
                        )

    val_loader = DataLoader(dset_val,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4
                        )
    print("DataLoader is loaded with {} number of worker and {} batch size.".format(4, batch_size))

    dataloaders = {}
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader
    
    epoch_losses = {"train":[], "val":[]}
    since = time.time()

    best_loss = 0.1

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        
        #print('Epoch {}/{}.....{}.....{}'.format(epoch, num_epochs - 1,batch_size, lr))
        #print('-' * 10)
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_losses[phase].append(epoch_loss)
                 
            print('Epoch {}/{}.....{}.....{}\n'.format(epoch, num_epochs - 1,batch_size, lr), '{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(),'C:\\Users\\deepd\\OneDrive\\Masaüstü\\Dataset\\Sequence\\test_1\\{}_{}_{}_loss_{:.4f}_PG.pth'.format(batch_size,weight_decay,epoch, epoch_loss))
                print("multi_{}_{}_{}_Pg.pth file is saved".format(batch_size,lr,epoch))
                  


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    #----------------------------------------------------------------------------------
    plt.figure()

    plt.plot(epoch_losses['train'], label = "Train Loss")
    plt.plot(epoch_losses['val'], label = "Val Loss")
    plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=.5)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    plt.title('Train')

    plt.show()
    plt.savefig("Sequence_321_multi{}_{}.png".format(batch_size,lr))
    #----------------------------------------------------------------------------------

    #return model


#-------------------------------------MultiProcess---------------------------------
if __name__ == '__main__':

    num_epochs = 12
    lr = [0.001,0.00001]
    batch_size = [32,128]
    weight_decay = 0.001



    mp.set_start_method('spawn')
    model = ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
    model.share_memory()
    #train_model(model, num_epochs, 16,0.001)
    
    processes = []
    
    for i in range(2):
        for j in range(2):
            print(("Train is starting batch_size: {} lr: {}"). format(batch_size[i], lr[j]))
            p = mp.Process(target = train_model, args = (model, num_epochs , batch_size[i], lr[j], weight_decay))
            p.start()
            processes.append(p)
    print("Our process:", processes)

    for p in processes:
        p.join()
    
    