# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:42:25 2022
被判断为正常（1）和自闭（0）的样本的概率落在哪些区间，这些区间之外的样本为不确定样本。
@author: R7-1700JY
这个程序没有用到论文中
"""

import torch
import math
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3_Large
from models.mobilenetv3 import MobileNetV3_Small


# 定义一些超参数 
batch_size = 64 
learning_rate = 0.005
num_epoches = 60

norm_mean=[0.485,0.456,0.406]
norm_std=[0.229,0.224,0.225]
data_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])

train_dataset = datasets.ImageFolder(root='./data/train',transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,drop_last=False,
                              num_workers=0) 

test_batchs=16
test_dataset = datasets.ImageFolder(root='./data/test',transform=data_transform)
test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=test_batchs,
                              shuffle=True,drop_last=False,
                              num_workers=0) 

if torch.cuda.is_available():
    use_cuda=True
    print("cuda is available!!!")

#选择要训练的模型     
modelstr="MobileNetV2"
if modelstr=="MobileNetV1" :
    model = MobileNetV1() 
if modelstr=="MobileNetV2" :
    model = MobileNetV2()    
elif modelstr=="MobileNetV3_Large" :
    model =MobileNetV3_Large() 
elif modelstr=="MobileNetV3_Samll" :
    model =MobileNetV3_Small() 
    
if use_cuda:
    model = model.cuda()
    
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
transfer_type="notransfer"


#导入预训练的模型参数
print("load pretrained model's paratmeters......")
transfer_type="transfer"
parameterfile='./pretrainedModels/'+modelstr+'.pth'
print("pretrainedmodel's file:"+parameterfile)
pretrained_dict=torch.load(parameterfile)
model_dict = model.state_dict()
state_dict={}
i=0
j=0
for k,v in pretrained_dict.items():
    i=i+1
    if (k in model_dict.keys()) and (k[0:k.index('.')]not in('classifier','linear','linear4','fc')):
        state_dict[k]=v
        j=j+1
        
model_dict.update(state_dict)
model.load_state_dict(model_dict)    

#定义一个数组用来装各区间的概率统计值，例如，1表示0.0-0.2之间的概率统计值。
max_per0_arry=[0]*11
max_per1_arry=[0]*11
per0_array=[0]*11
per1_array=[0]*11
uncertaincount=0 #不确定样本的数量
FN_array=[0]*11
FP_array=[0]*11

#定义一些指标变量
max_epoch=0
max_test_acc=0.0000 
max_test_acc_val=0.0000
max_sensitivity=0.0
max_specificity=0.0
max_G_Mean=0.0
max_error_rate=0.0
max_F_Measure=0.0
max_auc=0.0

#训练模型 
print("The model is:"+modelstr)  
print("Begin to train model...")
time1 = time.time() #记录模型开始训练时间
for epoch in range(num_epoches):
    per0_array=[0]*11
    per1_array=[0]*11

    TP_count=0
    TN_count=0  
    TP_array=[0]*11
    TN_array=[0]*11
    
    FN_count=0
    FP_count=0
    FN_array=[0]*11
    FP_array=[0]*11
    for i,(inputs,labels) in enumerate(train_dataloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()            
        _, predicted = torch.max(outputs.data, 1)

    #在测试集上评估模型
    model.eval()
    test_loss = 0.0000
    test_acc = 0.0000
    num_correct = 0
    row_i=0
    y=[9]*len(test_dataset) #len(test_dataset) 是测试集中包含图片的数量
    pred_per=[9.0000]*len(test_dataset)    
    TP=0
    TN=0
    FP=0
    FN=0
    count=0
    uncertaincount=0
    for data in test_dataloader:
        img, label = data
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = model(img)
        loss = criterion(out, label)
        test_loss += loss.data.item()*label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        test_acc += num_correct.item()
            
        #数组y是真实值，和标签数组是一样的。只是标签在一次循环中是test_batchs个数，要把它们放到y中
        percentage=torch.nn.functional.softmax(out,dim=1)
        count=count+1
        #print("count=",count," len(label)=",len(label),"row_i",row_i)
        if len(label)==test_batchs:
            for i in range(test_batchs):  #前面的批都够一个test_batchs，例如总数104，一个batchs是20，那么最后一个不够20个
                y[row_i*test_batchs+i]=label[i].item()
                pred_per[row_i*test_batchs+i]=round(percentage[i,1].item(),4)
                
                per0=math.floor(percentage[i,0].item()*10)
                per1=math.floor(percentage[i,1].item()*10)
                if round(percentage[i,0].item(),4)==round(percentage[i,1].item(),4):
                    uncertaincount+=1
                '''
                print('percentage[i,0].item()=',str(percentage[i,0].item()))
                print('percentage[i,1].item()=',str(percentage[i,1].item()))
                print('per0=',per0)
                print('per1=',per1)
                
                if per0>per1: #样本被判断为0(自闭症)的概率落到那个区间
                    per0_array[per0]+=1
                    print('per0_array[per0]+=1')
                elif per0<per1: #样本被判断为1(正常)的概率落到那个区间
                    per1_array[per1]+=1
                    print('per1_array[per1]+=1')
                else:  #样本两个概率一样大，是不确定样本
                    uncertaincount+=1
                '''
                
                if (label[i].item()==1 and pred[i].item()==1):
                    TP=TP+1
                    '''
                    print('TP,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('TP,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    TP_array[per1]+=1
                    TP_count+=1
                if (label[i].item()==0 and pred[i].item()==0):
                    TN=TN+1
                    '''
                    print('TN,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('TN,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    TN_array[per0]+=1
                    TN_count+=1                    
                if (label[i].item()==1 and pred[i].item()==0):
                    FN=FN+1
                    '''
                    print('FN,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('FN,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    FN_array[per0]+=1
                    FN_count+=1
                if (label[i].item()==0 and pred[i].item()==1):
                    FP=FP+1
                    '''
                    print('FP,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('FP,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    FP_array[per1]+=1
                    FP_count+=1
        else:
            for i in range(len(test_dataset)-row_i*test_batchs):#最后一个test_batchs的处理
                #print("len(test_dataset)-row_i*test_batchs=",len(test_dataset)-row_i*test_batchs)
                y[row_i*test_batchs+i]=label[i].item()
                pred_per[row_i*test_batchs+i]=round(percentage[i,1].item(),4)
                
                per0=math.floor(percentage[i,0].item()*10)
                per1=math.floor(percentage[i,1].item()*10)
                
                if (label[i].item()==1 and pred[i].item()==1):
                    TP=TP+1
                    TP_array[per1]+=1
                    TP_count+=1                    
                if (label[i].item()==0 and pred[i].item()==0):
                    TN=TN+1
                    TN_array[per0]+=1
                    TN_count+=1                    
                if (label[i].item()==1 and pred[i].item()==0):
                    FN=FN+1
                    '''
                    print('FN,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('FN,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    FN_array[per0]+=1
                    FN_count+=1
                if (label[i].item()==0 and pred[i].item()==1):
                    FP=FP+1
                    '''
                    print('FP,percentage[i,0].item()=',str(percentage[i,0].item()))
                    print('FP,percentage[i,1].item()=',str(percentage[i,1].item()))
                    print('per0=',per0)
                    print('per1=',per1)
                    '''
                    FP_array[per1]+=1
                    FP_count+=1
        
        row_i=row_i+1
        
    #每test一次，就计算一次sensitivity等指标
    sensitivity=TP/(TP+FN) #即TPR，可以通过求acc同时获得
    specificity=TN/(TN+FP) 
    G_Mean=(sensitivity*specificity)**0.5  #开平方根
    error_rate=(FP+FN)/(TP+TN+FP+FN)  #即FPR，可以通过求acc同时获得
    F_Measure=2*TP/(2*TP+FP+FN)
    fpr,tpr,thresholds=roc_curve(y,pred_per,pos_label=1)
    roc_auc = round(auc(fpr,tpr),4)
    test_acc=test_acc /len(test_dataset) #当前epoch的测试准确率

    if test_acc>max_test_acc: 
        max_test_acc=test_acc  #当前测试准确率取代最大测试准确率
        max_epoch=epoch
        max_sensitivity=sensitivity
        max_specificity=specificity
        max_error_rate=error_rate
        max_auc=roc_auc
        max_G_Mean=G_Mean
        max_F_Measure=F_Measure
        max_per0_arry=per0_array
        max_per1_arry=per1_array
        
        #这句话用来保存模型
        if max_test_acc<=0.83 and max_auc<=0.85:           
            torch.save(model,'./bestmodels/tmp/'+transfer_type+'-'+modelstr+'('+str(round(max_test_acc,4))+"+"+str(round(max_auc,4))+')'+'.pth')
            #break

    #输出阳性概率和阴性概率的区间统计
    print('epoch:',epoch)
    print("per0_array:",str(per0_array))
    print("per1_array:",str(per1_array))
    print('TP_count=',str(TP_count))
    print("TP_array:",str(TP_array))  
    print('TN_count=',str(TN_count))
    print("TN_array:",str(TN_array))    
    print('FN_count=',str(FN_count))
    print("FN_array:",str(FN_array))
    print('FP_count=',str(FP_count))
    print("FP_array:",str(FP_array))
    print('uncertaincount=',uncertaincount)
            
print("max_epoch is:",max_epoch)
print("max_test_acc is:",max_test_acc)
print("max_auc is:",max_auc)    
print("max_sensitivity is:",max_sensitivity)    
print("max_specificity is:",max_specificity)  
print("max_error_rate is:",max_error_rate)
print("max_G_Mean is:",max_G_Mean)   
print("max_F_Measure is:",max_F_Measure)
print("max_per0_array:",str(per0_array))
print("max_per1_array:",str(per1_array))

time2 = time.time()
#输出整数，秒数 
print("Time spent is:"+str(int(time2-time1)))


