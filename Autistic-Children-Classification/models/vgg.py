# -*- coding: utf-8 -*-
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#VGG网络的输入为224*224*3的图片
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
} #以VGG16为例，64,64表示经过2次卷积（卷积核为64个），M表示池化，256,256,256表示经过3次卷积（卷积核为256个）


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.bn=nn.BatchNorm1d(512)
        self.relu=nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512,2) #如果是44*44的图像，用nn.Linear(512, 2)；如果是224*224，那就不是这样了。
        '''
        self.classifier = nn.Linear(512,256)
        
        self.bn1=nn.BatchNorm1d(256)
        self.classifier1 = nn.Linear(256, 128)
        
        self.bn2=nn.BatchNorm1d(128)
        self.classifier2 = nn.Linear(128, 2)
        '''
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.classifier(out)
        '''
        out = self.bn(out)
        out = self.classifier(out)
        out = self.bn1(out)
        out = self.classifier1(out)
        out = self.bn2(out)
        out = self.classifier2(out)
        '''
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #layers += [nn.AvgPool2d(7)] #如果图片是224*224，就加这一句，意思是把7*7*512经过7*7卷积核平均池化后变成1*1*512，否则出现矩阵16*25088无法和512*2相乘的错误
        return nn.Sequential(*layers)

