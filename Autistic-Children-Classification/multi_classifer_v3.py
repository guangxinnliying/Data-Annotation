# -*- coding: utf-8 -*-
#三个分类器整合成一个新的分类器，使用投票方法。
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
from models.mobilenetv1 import MobileNetV1
from models.vgg import VGG


# 定义一些超参数
batch_size = 64 

norm_mean=[0.485,0.456,0.406]
norm_std=[0.229,0.224,0.225]
data_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])

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
#recordfile='./bestmodels/10timerecord.txt'
#file=open(recordfile,'w')

total_test_acc1 =0.0  
total_sensitivity1=0.0
total_specificity1=0.0  
total_error_rate1=0.0  
total_roc_auc1=0.0 
total_G_Mean1=0.0
total_F_Measure1=0.0

total_test_acc2 =0.0  
total_sensitivity2=0.0
total_specificity2=0.0  
total_error_rate2=0.0  
total_roc_auc2=0.0 
total_G_Mean2=0.0
total_F_Measure2=0.0

total_test_acc3 =0.0  
total_sensitivity3=0.0
total_specificity3=0.0  
total_error_rate3=0.0  
total_roc_auc3=0.0 
total_G_Mean3=0.0
total_F_Measure3=0.0

total_test_acc4 =0.0  
total_sensitivity4=0.0
total_specificity4=0.0  
total_error_rate4=0.0  
total_roc_auc4=0.0 
total_G_Mean4=0.0
total_F_Measure4=0.0
    
for current_time in range(total_time):
    modelstr1="MobileNetV2"
    modelstr2="MobileNetV3-Large"
    modelstr3="MobileNetV3-Small"
    model1=torch.load('./bestmodels/'+modelstr1+'T'+str(current_time)+'.pth')
    model2=torch.load('./bestmodels/'+modelstr2+'T'+str(current_time)+'.pth')
    model3=torch.load('./bestmodels/'+modelstr3+'T'+str(current_time)+'.pth')
    #model4=torch.load('./bestmodels/VGG19T'+str(current_time)+'.pth')

    row_i=0

    y=[9]*len(test_dataset) #len(test_dataset) 是测试集中包含图片的数量
    #下面的变量是MoblieNetV1的
    pred_per1=[9.]*len(test_dataset)    
    TP1=0
    TN1=0
    FP1=0
    FN1=0
    #下面的变量是MoblieNetV2的
    pred_per2=[9.]*len(test_dataset)    
    TP2=0
    TN2=0
    FP2=0
    FN2=0
    #下面的变量是VGG16的
    pred_per3=[9.]*len(test_dataset)    
    TP3=0
    TN3=0
    FP3=0
    FN3=0
    #下面的变量是MoblieNetV1+MoblieNetV2+VGG16的
    pred_per4=[9.]*len(test_dataset)    
    TP4=0
    TN4=0
    FP4=0
    FN4=0
    count=0
    
    #下面的变量用来统计不同投票情况下的TP,TN,FP,FN
    case1_TP4=0
    case1_TN4=0
    case1_FP4=0
    case1_FN4=0
    
    case2_TP4=0
    case2_TN4=0
    case2_FP4=0
    case2_FN4=0
    
    case3_TP4=0
    case3_TN4=0
    case3_FP4=0
    case3_FN4=0
    
    #下面几个变量是用标记或来统计不同投票情况下的TP,TN,FP,FN等情况
    case1_flag=0
    case2_flag=0
    case3_flag=0
    
    for data in test_dataloader:
        img, label = data
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out1 = model1(img)
        _, pred_label1 = torch.max(out1, 1)
        
        out2 = model2(img)
        _, pred_label2 = torch.max(out2, 1)
        
        out3 = model3(img)
        _, pred_label3 = torch.max(out3, 1)
        '''
        out4 = model4(img)
        _, pred_label4 = torch.max(out4, 1)
        '''
        #数组y是真实值，和标签数组是一样的。只是标签在一次循环中是test_batchs个数，要把它们放到y中
        percentage1=torch.nn.functional.softmax(out1,dim=1)
        percentage2=torch.nn.functional.softmax(out2,dim=1)
        percentage3=torch.nn.functional.softmax(out3,dim=1)
        #percentage4=torch.nn.functional.softmax(out4,dim=1)
        if len(label)==test_batchs:
            endval=test_batchs
        else:
            endval=len(test_dataset)-row_i*test_batchs
        
        case1_flag=0
        case2_flag=0
        case3_flag=0
        
        for i in range(endval):  #前面的批都够一个test_batchs，例如总数104，一个batchs是20，那么最后一个不够20个
            y[row_i*test_batchs+i]=label[i].item()
            per1=round(percentage1[i,1].item(),4) #mobilenetv1的正类概率
            per2=round(percentage2[i,1].item(),4) #mobilenetv2的正类概率
            per3=round(percentage3[i,1].item(),4) #VGG16的正类概率
            #per4=round(percentage4[i,1].item(),4) #VGG19的正类概率
            per5=0.0     
            sum_pre_label=int(pred_label1)+int(pred_label2)+int(pred_label3)
            if (sum_pre_label==0 or sum_pre_label==3): #若三个模型的预测标签一样，则取二者预测概率的平均值
                per4=round(max(per1,per2,per3),4)
                pred_label4=pred_label1[i].item()            
                case1_flag=1
            elif sum_pre_label==1: #若3个模型的预测标签有2个为0,1个为1的情况。
                pred_label4=0
                case2_flag=1
                if int(pred_label1)==1:
                    per4=round((per2+per3)/2,4)

                elif int(pred_label2)==1:
                    per4=round((per1+per3)/2,4)

                elif int(pred_label3)==1:
                    per4=round((per1+per2)/2,4)
 
            elif sum_pre_label==2:  #若3个模型的预测标签有2个为1,1个为0的情况。
                pred_label4=1
                case3_flag=1
                if int(pred_label1)==0:
                    per4=round((per2+per3)/2,4)

                elif int(pred_label2)==0:
                    per4=round((per1+per3)/2,4)

                elif int(pred_label3)==0:
                    per4=round((per1+per2)/2,4)
                    #pred_label4=pred_label1[1].item()

            pred_per1[row_i*test_batchs+i]=round(per1,4)
            pred_per2[row_i*test_batchs+i]=round(per2,4)
            pred_per3[row_i*test_batchs+i]=round(per3,4)
            pred_per4[row_i*test_batchs+i]=round(per4,4)
            
        
            if (label[i].item()==1 and pred_label1[i].item()==1):
                TP1=TP1+1
            if (label[i].item()==0 and pred_label1[i].item()==0):
                TN1=TN1+1
            if (label[i].item()==1 and pred_label1[i].item()==0):
                FN1=FN1+1
            if (label[i].item()==0 and pred_label1[i].item()==1):
                FP1=FP1+1
            
            if (label[i].item()==1 and pred_label2[i].item()==1):
                TP2=TP2+1
            if (label[i].item()==0 and pred_label2[i].item()==0):
                TN2=TN2+1
            if (label[i].item()==1 and pred_label2[i].item()==0):
                FN2=FN2+1
            if (label[i].item()==0 and pred_label2[i].item()==1):
                FP2=FP2+1
            
            if (label[i].item()==1 and pred_label3==1):
                TP3=TP3+1
            if (label[i].item()==0 and pred_label3==0):
                TN3=TN3+1
            if (label[i].item()==1 and pred_label3==0):
                FN3=FN3+1
            if (label[i].item()==0 and pred_label3==1):
                FP3=FP3+1
                
            if case1_flag==1:  #四个分类器都给出相同的标签
                if (label[i].item()==1 and pred_label4==1):
                    case1_TP4+=1
                if (label[i].item()==0 and pred_label4==0):
                    case1_TN4+=1
                if (label[i].item()==1 and pred_label4==0):
                    case1_FN4+=1
                if (label[i].item()==0 and pred_label4==1):
                    case1_FP4+=1 

            if case2_flag==1:  #四个分类器中，三个给0,1个给1
                if (label[i].item()==1 and pred_label4==1):
                    case2_TP4+=1
                if (label[i].item()==0 and pred_label4==0):
                    case2_TN4+=1
                if (label[i].item()==1 and pred_label4==0):
                    case2_FN4+=1
                if (label[i].item()==0 and pred_label4==1):
                    case2_FP4+=1
                    
            if case3_flag==1:  #四个分类器中，三个给1,1个给0
                if (label[i].item()==1 and pred_label4==1):
                    case3_TP4+=1
                if (label[i].item()==0 and pred_label4==0):
                    case3_TN4+=1
                if (label[i].item()==1 and pred_label4==0):
                    case3_FN4+=1
                if (label[i].item()==0 and pred_label4==1):
                    case3_FP4+=1 
               

                
        row_i=row_i+1
        #print("TP1:",TP1,"fn1:",FN1)
    #每评估一次，就计算一次sensitivity等指标    
    sensitivity1=TP1/(TP1+FN1) #即TPR，可以通过求acc同时获得
    specificity1=TN1/(TN1+FP1) 
    G_Mean1=(sensitivity1*specificity1)**0.5  #开平方根
    error_rate1=(FP1+FN1)/(TP1+TN1+FP1+FN1)  #即FPR，可以通过求acc同时获得
    F_Measure1=2*TP1/(2*TP1+FP1+FN1)
    fpr1,tpr1,thresholds1=roc_curve(y,pred_per1,pos_label=1)
    roc_auc1 = round(auc(fpr1,tpr1),4)
    test_acc1=1-error_rate1 

    sensitivity2=TP2/(TP2+FN2) #即TPR，可以通过求acc同时获得
    specificity2=TN2/(TN2+FP2) 
    G_Mean2=(sensitivity2*specificity2)**0.5  #开平方根
    error_rate2=(FP2+FN2)/(TP2+TN2+FP2+FN2)  #即FPR，可以通过求acc同时获得
    F_Measure2=2*TP2/(2*TP2+FP2+FN2)
    fpr2,tpr2,thresholds2=roc_curve(y,pred_per2,pos_label=1)
    roc_auc2 = round(auc(fpr2,tpr2),4)
    test_acc2=1-error_rate2 #当前epoch的测试准确率

    sensitivity3=TP3/(TP3+FN3) #即TPR，可以通过求acc同时获得
    specificity3=TN3/(TN3+FP3) 
    G_Mean3=(sensitivity3*specificity3)**0.5  #开平方根
    error_rate3=(FP3+FN3)/(TP3+TN3+FP3+FN3)  #即FPR，可以通过求acc同时获得
    F_Measure3=2*TP3/(2*TP3+FP3+FN3)
    fpr3,tpr3,thresholds3=roc_curve(y,pred_per3,pos_label=1)
    roc_auc3 = round(auc(fpr3,tpr3),4)
    test_acc3=1-error_rate3 #当前epoch的测试准确率
    
    TP4=case1_TP4+case2_TP4+case3_TP4
    TN4=case1_TN4+case2_TN4+case3_TN4
    FP4=case1_FP4+case2_FP4+case3_FP4
    FN4=case1_FN4+case2_FN4+case3_FN4
    
    sensitivity4=TP4/(TP4+FN4) #即TPR，可以通过求acc同时获得
    specificity4=TN4/(TN4+FP4) 
    G_Mean4=(sensitivity4*specificity4)**0.5  #开平方根
    error_rate4=(FP4+FN4)/(TP4+TN4+FP4+FN4)  #即FPR，可以通过求acc同时获得
    F_Measure4=2*TP4/(2*TP4+FP4+FN4)
    fpr4,tpr4,thresholds4=roc_curve(y,pred_per4,pos_label=1)
    roc_auc4 = round(auc(fpr4,tpr4),4)
    test_acc4=1-error_rate4 #当前epoch的测试准确率
    
    print("Evaluation index of "+modelstr1+":")    
    print("test_acc is:",test_acc1)    
    print("sensitivity is:",sensitivity1) 
    print("specificity is:",specificity1)    
    print("error_rate is:",error_rate1)  
    print("auc is:",roc_auc1)    
    print("G_Mean is:",G_Mean1)    
    print("F_Measure is:",F_Measure1) 

    print("Evaluation index of "+modelstr2+":")     
    print("test_acc is:",test_acc2)    
    print("sensitivity is:",sensitivity2) 
    print("specificity is:",specificity2)    
    print("error_rate is:",error_rate2)  
    print("auc is:",roc_auc2)    
    print("G_Mean is:",G_Mean2)    
    print("F_Measure is:",F_Measure2) 

    print("Evaluation index of "+modelstr3+":")     
    print("test_acc is:",test_acc3)    
    print("sensitivity is:",sensitivity3) 
    print("specificity is:",specificity3)    
    print("error_rate is:",error_rate3)  
    print("auc is:",roc_auc3)    
    print("G_Mean is:",G_Mean3)    
    print("F_Measure is:",F_Measure3)  
    
    print("Evaluation index of "+modelstr1+"&"+modelstr2+"&"+modelstr3+":")     
    print("test_acc is:",test_acc4)    
    print("sensitivity is:",sensitivity4) 
    print("specificity is:",specificity4)    
    print("error_rate is:",error_rate4)  
    print("auc is:",roc_auc4)    
    print("G_Mean is:",G_Mean4)    
    print("F_Measure is:",F_Measure4)      

    print("count:",count)



   
print('case 1: TP4=',case1_TP4,'  TN4=',case1_TN4,'  FP4=',case1_FP4,'  FN5=',case1_FN4)
print('case 2: TP4=',case2_TP4,'  TN4=',case2_TN4,'  FP4=',case2_FP4,'  FN5=',case2_FN4)
print('case 3: TP4=',case3_TP4,'  TN4=',case3_TN4,'  FP4=',case3_FP4,'  FN5=',case3_FN4)





