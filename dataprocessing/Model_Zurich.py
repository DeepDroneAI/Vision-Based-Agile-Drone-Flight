import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=True)
        self.max_pool = nn.MaxPool2d(3, stride= 2)
        self.bn1 = nn.BatchNorm2d(32)
         
        #First Residual Block
        self.conv2 = nn.Conv2d(32,32, kernel_size = 3, stride = 2, padding = 1, bias = True)
        self.drop1 = nn.Dropout2d(0.5)
        self.conv2a = nn.Conv2d(32,32, kernel_size = 1, stride = 2, padding= 0, bias=True)
        self.conv3 = nn.Conv2d(32,32, kernel_size = 3, stride = 2, padding = 13, bias = True)

        #Second Residual Block
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(32,64, kernel_size = 3, stride = 2, padding = 1, bias = True)
        self.conv4a = nn.Conv2d(32,64, kernel_size = 1, stride = 2, padding= 0, bias=True)
        self.conv5 = nn.Conv2d(64,64, kernel_size = 3, stride = 2, padding = 7, bias = True)
        
        #Third Residual Block
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(64,128, kernel_size = 3, stride = 2, padding = 1, bias = True)
        self.conv6a = nn.Conv2d(64,128, kernel_size = 1, stride = 2, padding= 0, bias=True)
        self.conv7 = nn.Conv2d(128,128, kernel_size = 3, stride = 2, padding = 4, bias = True)    
        #self.pool = nn.AvgPool2d(3)
        self.linear = nn.Linear(6272, 4)
        self.linear2 = nn.Linear(128,32)
        self.linear3 = nn.Linear(32,4)
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        #self.padding = torch.nn.ZeroPad2d((0, 0, 1, 0))
    def forward(self, x):
        #print(x.shape)

        x1 = self.conv1(x)
        #print("x1_shape:", x1.shape)
        x1 = self.max_pool(x1)
        #print("x1_shape:", x1.shape)
        x1 = F.relu(self.bn1(x1))
        
        x1 = self.drop1(x1)
        #First Residual Block
        x2 = F.relu(self.bn1(self.conv2(x1)))
        x2 = self.drop1(x2)
        #print("x2_shape:", x2.shape)
        x2 = self.conv3(x2)
        #print("x2_shape:", x2.shape)
        xa2 =self.conv2a(x1)
        #print("xa2_shape:",xa2.shape)
        x3 = x2 + xa2
        #print("x3_shape:", x3.shape)
        x3 = self.drop1(x3)

        x3 = F.relu(self.bn1(x3))

        x3 = self.drop1(x3)
        #Second Residual Block
        x4 = F.relu(self.bn2(self.conv4(x3)))
        x4 = self.drop1(x4)
        x4 = self.conv5(x4)
        x4 = self.drop1(x4)
        xa4 = self.conv4a(x3)

        x5 = x4 + xa4
        x5 = F.relu(self.bn2(x5))
    
        x5 = self.drop1(x5)
        #Third Residual Block
        x6 = F.relu(self.bn3(self.conv6(x5)))
        x6 = self.drop1(x6)
        x6 = self.conv7(x6)
        
        xa6 = self.conv6a(x5)
        
        x7 = x6 + xa6
    
        out = x7.view(x7.size(0), -1)
        out = F.relu(out)
        
        out = self.dropout1(out)
        steer = self.linear(out)
        
        
        return steer
