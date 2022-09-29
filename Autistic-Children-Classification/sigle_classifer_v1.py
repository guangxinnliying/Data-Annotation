# -*- coding: utf-8 -*-
#不确定性机器学习，单个分类器
#根据区间概率进行判断，不在指定区间的样本为不确定样本。
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3_Large

# 定义一些超参数
batch_size = 64 

norm_mean=[0.485,0.456,0.406]
norm_std=[0.229,0.224,0.225]
data_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])

samplefile="./data/samplelist_train.txt"  #用于训练及验证
samplefile_test="./data/samplelist_test.txt" #用于评估
test_batchs=1
total_time=1
if torch.cuda.is_available():
    use_cuda=True
    print("cuda is available!!!")


test_dataset = datasets.ImageFolder(root='./data/test',transform=data_transform)
test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=test_batchs,
                              shuffle=True,drop_last=False,
                              num_workers=0) 
#recordfile='./bestmodels/resultrecord.txt'
#file=open(recordfile,'w')

current_time=0
modelname="MobileNetV1"
model=torch.load('./bestmodels/'+modelname+'T'+str(current_time)+'.pth')

row_i=0

pred_per1=[9.]*len(test_dataset)    
TP=0
TN=0
FP=0
FN=0
count1=0
count2=0

for data in test_dataloader:
    img, label = data
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    print(img.shape)

    out1 = model(img)
    _, pred_label = torch.max(out1, 1)

            
    percentage=torch.nn.functional.softmax(out1,dim=1)
    print('percentage.shape:'+str(percentage.shape))
    print(percentage)


    tmp1=round(percentage[0,0].item(),4)
    tmp2=round(percentage[0,1].item(),4)  
    pred_label_tmp=0
    unclearflag=0
    
    if tmp1>tmp2:
        pred_label_tmp=0
    elif tmp1<tmp2:
        pred_label_tmp=1
    else: #阳性和阴性的概率相等，无法判断是阳性还是阴性，所以是不确定样本
        count1+=1
        
        
    if (label.item()==1 and pred_label.item()==1):
        TP=TP+1
    if (label.item()==0 and pred_label.item()==0):
        TN=TN+1
    if (label.item()==1 and pred_label.item()==0):
        FN=FN+1
        count1+=1
    if (label.item()==0 and pred_label.item()==1):
        FP=FP+1
        count1+=1
    
    #下面的if语句块是用于基于概率区间统计得到不确定样本
    if tmp1>tmp2:
        if tmp1>=0.9 and tmp1<=1:
            pred_label_tmp=0
        else:  #这个概率值不在大多数被正确判断的样本所在区间，所以样本是不确定样本
            count2+=1
    elif tmp1<tmp2:
        if tmp2>=0.9 and tmp2<=1:
            pred_label_tmp=1
        else: #这个概率值不在大多数被正确判断的样本所在区间，所以样本是不确定样本
            count2+=1
    else: #阳性和阴性的概率相等，无法判断是阳性还是阴性，所以是不确定样本
        count2+=1
        
        
    row_i=row_i+1

print('model: '+modelname)
print('the number samples of the right lable,TP='+str(TP)+'   TN='+str(TN))
print('the number samples of the wrong lable,FP='+str(FP)+'  FN='+str(FN))
print("unclear sample I:",str(count1))
print("unclear sample II:",str(count2))







