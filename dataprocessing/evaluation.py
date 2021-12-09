
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#################################Import-Libraries####################################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from dataprocessing import DatasetProcessing
from Model_Zurich import ResNet
from Model import ResNet8, BasicBlock
import matplotlib.pyplot as plt
from collections import OrderedDict
from Dronet import ResNet, BasicBlock

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ###################################################################################

    #################################DataPath##########################################
    print (os.getcwd())

    data_path = "/home/recep/dataprocessing/data"

    test_data = 'test'

    test_label_file = 'test_labels.txt'

    test_image_file = 'test_img.txt'

    print("###############DataPath is given.#################")
    ################################################################################

    ##############################Import-Dataset####################################
    transformations = transforms.Compose([
            transforms.Resize([200, 200]),
            transforms.ToTensor()],
            )

    dset_test = DatasetProcessing(
        data_path, test_data, test_image_file, test_label_file, transformations)

    
    image_datasets = {}
    image_datasets["test"] = dset_test
    #print(image_datasets["val"])
    #image_datasets["test"] = dset_test
    dataset_sizes = {}
    dataset_sizes["test"] = len(image_datasets["test"])
    print("Test %d images" % (
        dataset_sizes["test"] ))
    #######################################################################################

    ##################################DataLoader##########################################
    #Initilized our parameters
    batch_size = 1
    num_workers = 4


    test_loader = DataLoader(dset_test,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )

    print("DataLoader is loaded with {} number of worker and {} batch size.".format(num_workers, batch_size))


    dataloaders = {}
    dataloaders["test"] = test_loader


    #######################################################################################
    prediction = {"pred":[], "true":[]}
    criterion = nn.MSELoss()

    model_ft =  ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
    #print(model_ft)
    print("Model is Updated.")

    model = model_ft.to(device)
    #model.load_state_dict(torch.load('/home/recep/dataprocessing/model/16_0.1_26_loss_0.0413_PG.pth'))   
    #model.load_state_dict(torch.load('/home/recep/dataprocessing/model/16_0.001_2_loss_0.0101_PG.pth'))  ### İYİ SONUÇ 
    model.load_state_dict(torch.load('/home/recep/DeepDrone-Traj/weights/Dronet_new.pth')) 


    #model_ft2 =  ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
    #print(model_ft)
    #print("Model is Updated.")

    #model2 = model_ft2.to(device)
    #model2.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\DeepDrone\\newlittle\model\\test_6\\16_0.0001_10_loss_0.0137_PG.pth'))   

    #model.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\newlittle\model\\test_4\\16_0.0001_4_loss_0.0199_PG.pth'))   
    #model.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\newlittle\model\\test_6\\16_0.0001_10_loss_0.0137_PG.pth'))   
    #model2.eval()

    images_so_far = 0
    pred_depth = []
    true_depth = []
    true_angle1 = []
    true_angle2 = []
    true_angle3 = []
    pred_angle1 = []
    pred_angle2 = []
    pred_angle3 = []
    exit_list = []
    running_loss = 0.0
    with torch.no_grad():
        
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels=labels.reshape(-1,1)
            a = labels[2] 
            b = labels[1]
           
           
            labels = torch.from_numpy(np.array([labels[0],b,a,labels[3]])).to(device)
            #labels = labels.to(device)
            labels=labels.reshape(-1,1)
            
        
            
            true_depth.append(labels[0])
            true_angle1.append(labels[1])
            true_angle2.append(labels[2])
            true_angle3.append(labels[3])

            
            
            outputs = model(inputs)
            outputs = outputs.reshape(-1,1)
            print("Outputs: ", outputs, "\n")
            print("Labels: ", labels, "\n")
             

            
            pred_depth.append(outputs[0])
            pred_angle1.append(outputs[1])
            pred_angle2.append(outputs[2])
            pred_angle3.append(outputs[3])


            
            loss = criterion(outputs, labels)
            

            running_loss += loss * batch_size
           
            #print("loss:",loss, i)
            """if loss > 1:
                exit_list.append(i)
                print("loss:",loss, dset_test.images[i])
                #print("exit_index:", i)
            else:
                #print(dset_test.label[i], dset_test.images[i])"""
            """f = open("new_label.txt","a")
                f.write( str(labels[0].item()) + " " + str(labels[1].item()) + " " + str(labels[2].item()) +  " " +str(labels[3].item())+ '\n')
                f.close()  

                f = open("new_img.txt","a")
                f.write(str(dset_test.images[i].item())+ " " + str(dset_test.images[i]) + '\n')
                f.close()""" 
            
            """true = open("truefor_R.txt", "a")
            true.write(str(labels[0].item())+ " " + str(labels[1].item()) + " " + str(labels[2].item()) + " " +str(labels[3].item())+ '\n')
            true.close()"""
            
            pred = open("predfor_Rtest.txt", "a")
            pred.write(str(outputs[0].item()) + " " + str(outputs[1].item()) + " " + str(outputs[2].item()) + " " + str(outputs[3].item()) +'\n')
            
            error = np.abs(np.array(labels.to("cpu") - outputs.to("cpu")))
            
            Mea_cov =  np.dot(error,error.T)
            Mea_cov = np.diag(Mea_cov)
            
            
            cov = open("cov_Rtest.txt","a")
            cov.write(str(Mea_cov[0].item()) + " " +str(Mea_cov[1].item()) + " " +str(Mea_cov[2].item()) + " " +str(Mea_cov[3].item()) + " "  '\n')
            cov.close()

            """
            f = open("input_valueforLSTMf.txt","a")
            f.write(str(outputs[0].item()) + " " + str(outputs[1].item()) + " " + str(outputs[2].item()) + " " + str(outputs[3].item()) + '\n')
            f.close()"""

            """f = open("body2.txt","a")
            f.write("predicted values:"+  str(outputs[0].item()) + " " + str(outputs[1].item()) + " " + str(outputs[2].item()) + " " + str(outputs[3].item()) + '\n'
                    + "true values     :" +    str(labels[0].item()) + " " + str(labels[1].item()) + " " + str(labels[2].item()) +  " " +str(labels[3].item())+ '\n')
            f.close()"""
            

    epoch_loss = running_loss / dataset_sizes["test"]
    print(epoch_loss)        

    error1 = np.abs(np.array(true_depth) - np.array(pred_depth))*100
    error4 = np.abs(np.array(true_angle3) - np.array(pred_angle3))*57.2957
    error3 = np.abs(np.array(true_angle2) - np.array(pred_angle2))*57.2957
    error2 = np.abs(np.array(true_angle1) - np.array(pred_angle1))*57.2957

    print( "Test {}'lik set üzerinden yapılmıştır ve santimetre-derece cinsinden verilmektedir.".format(dataset_sizes["test"]))
    print("Ortalama Derinlik Hatası:", np.sum(error1)/dataset_sizes["test"])
    print("Ortalama spehrical Açı hataları:", np.sum(error2)/dataset_sizes["test"],np.sum(error3)/dataset_sizes["test"])
    print("Ortalama respective yaw açısı hatası:", np.sum(error4)/dataset_sizes["test"])

    print("Maximum errorlar sırasıyla:", np.max(error1), np.max(error2), np.max(error3), np.max(error4))
    #######################################################################################
    """
    plt.figure(figsize = (14.0,10.0))

    plt.plot(pred_depth[:400],label="Predictions")
    plt.plot(true_depth[:400], label="Actual Labels")

    plt.title('Depth Comparison btw True-Prediction', fontsize=14)
    plt.xlabel('data')
    plt.ylabel('Angles [Degree]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid('on')

    plt.show()
    plt.savefig("test_depth.png")  
    #######################################################################################
    plt.figure(figsize = (14.0,10.0))

    plt.plot(pred_angle1[:400],label="Predictions")
    plt.plot(true_angle1[:400], label="Actual Labels")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)


    plt.title('S-angle1 Comparison btw True-Prediction', fontsize=14)
    plt.xlabel('data')
    plt.ylabel('Angles [Degree]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid('on')

    plt.show()
    plt.savefig("test_a1.png")
    #######################################################################################
    plt.figure(figsize = (14.0,10.0))

    plt.plot(pred_angle2[:400],label="Predictions")
    plt.plot(true_angle2[:400], label="Actual Labels")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)


    plt.title('S-angle2 Comparison btw True-Prediction', fontsize=14)
    plt.xlabel('data')
    plt.ylabel('Angles [Degree]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid('on')

    plt.show()
    plt.savefig("test_a2.png") 
    #######################################################################################
    plt.figure(figsize = (14.0,10.0))

    plt.plot(pred_angle3[:400],label="Predictions")
    plt.plot(true_angle3[:400], label="Actual Labels")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)


    plt.title('Respective Yaw Angle Comparison btw True-Prediction', fontsize=14)
    plt.xlabel('data')
    plt.ylabel('Angles [Degree]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid('on')

    plt.show()
    plt.savefig("test_a3.png")




    #-----------------Subplot denemeleri--------------------
    t = range(500)

    plt, axs = plt.subplots(4, 1, figsize = (15.0,12.0))
    #plt.subplots_adjust(wspace=5.0)

    axs[0].plot(t,error1[:500], '-')
    #axs[0].set_title('subplot 1')
    #axs[0].set_xlabel('distance (m)')
    axs[0].set_ylabel('Error of Depth [cm]', fontsize = 13)
    plt.suptitle('Error of All states', fontsize=16)

    axs[1].plot(t,error2[:500], '-')
    #axs[1].set_xlabel('time (s)')
    #axs[1].set_title('subplot 2')
    axs[1].set_ylabel('Error of S-angle [o]',fontsize = 13)

    axs[2].plot(t,error3[:500], '-')
    #axs[2].set_title('subplot 1')
    #axs[2].set_xlabel('distance (m)')
    axs[2].set_ylabel('Error of S-angle [o]',fontsize = 13)


    axs[3].plot(t,error4[:500], '-')
    #axs[3].set_xlabel('time (s)')
    #axs[3].set_title('subplot 2')   
    axs[3].set_ylabel('Error of Respective Yaw Angle [o]',fontsize = 12)

    plt.show()
    plt.savefig("all_error.png") 

    #---------------------------------------------------------
    """