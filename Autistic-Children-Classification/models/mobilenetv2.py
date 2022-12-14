# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        # 通过 expansion 增大 feature map 的数量
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # 步长为 1 时，如果 in 和 out 的 feature map 通道不同，用一个卷积改变通道数
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes))
        # 步长为 1 时，如果 in 和 out 的 feature map 通道相同，直接返回输入
        if stride == 1 and in_planes == out_planes:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 步长为1，加 shortcut 操作
        if self.stride == 1:
            return out + self.shortcut(x)
        # 步长为2，直接输出
        else:
            return out
        
class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1), 
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes) #原先的分类层
        '''
        #新加的分类层
        
        self.myfc1 = nn.Linear(1280, 640)
        
        self.mybn2=nn.BatchNorm1d(640)
        self.myfc2 = nn.Linear(640, 320)

        
        self.mybn3=nn.BatchNorm1d(320)
        self.myfc3 = nn.Linear(320, 160)
        
        self.mybn4=nn.BatchNorm1d(160)
        self.myfc4 = nn.Linear(160, 80)        
        
        self.mybn5=nn.BatchNorm1d(80)
        self.myfc5 = nn.Linear(80, 40)
        
        self.mybn6=nn.BatchNorm1d(40)
        self.myfc6 = nn.Linear(40, 2)
        '''
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            #第一个ResidualBlock的步幅由make_layer的函数参数stride指定
            #后续的num_blocks-1个ResidualBlock步幅是1
            strides = [stride] + [1]*(num_blocks-1)
            '''
            [1]
			[1, 1]
			[2, 1, 1]
			[2, 1, 1, 1]
			[1, 1, 1]
			[2, 1, 1]
			[1]
            '''
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) #原来的分类层
        '''
        out = self.myfc1(out)
        out = self.mybn2(out)
        out = self.myfc2(out)
        out = self.mybn3(out)
        out = self.myfc3(out)
        out = self.mybn4(out)
        out = self.myfc4(out)
        out = self.mybn5(out)
        out = self.myfc5(out)
        out = self.mybn6(out)
        out = self.myfc6(out)
        '''
        return out