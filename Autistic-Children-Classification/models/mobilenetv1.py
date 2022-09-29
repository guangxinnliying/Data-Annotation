# -*- coding: utf-8 -*-

import torch.nn as nn
 
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
 
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
 
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
 
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(2) #44*44图像最后卷积得到2*2*1024矩阵，所以要用2来池化，变成1*1*1024
        )
        self.fc1 = nn.Linear(1024, 2)
        '''
        self.bn1=nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        
        self.bn2=nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)

        
        self.bn3=nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        
        self.bn4=nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)        
        
        self.bn5=nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        
        self.bn6=nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 2)
        '''
 
    def forward(self, x):
        x = self.model(x)  #这一句出错
        x = x.view(x.size(0), -1)
        x = self.fc1(x) 
        '''
        x = self.bn1(x)
        x = self.fc1(x) 
        x = self.bn2(x)
        x = self.fc2(x)        

        x = self.bn3(x)

        x = self.fc3(x)
        x = self.bn4(x)
        x = self.fc4(x)
        
        x = self.bn5(x)
        x = self.fc5(x)
        
        x = self.bn6(x)
        x = self.fc6(x)
        '''
        return x