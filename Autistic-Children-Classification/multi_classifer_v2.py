# -*- coding: utf-8 -*-
#两个分类器整合， 给出参与分类器及整合分类器的TN和FN
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
    
for current_time in range(total_time):
    modelname1="MobileNetV3-Large"
    modelname2="MobileNetV3-Small"
    model1=torch.load('./bestmodels/'+modelname1+'T'+str(current_time)+'.pth')
    model2=torch.load('./bestmodels/'+modelname2+'T'+str(current_time)+'.pth')

    row_i=0

    y=[9]*len(test_dataset) #len(test_dataset) 是测试集中包含图片的数量
    #下面的变量是model1的
    pred_per1=[9.]*len(test_dataset)    
    TP1=0
    TN1=0
    FP1=0
    FN1=0
    #下面的变量是model2的
    pred_per2=[9.]*len(test_dataset)    
    TP2=0
    TN2=0
    FP2=0
    FN2=0
    #下面的变量是model1+model2的
    pred_per3=[9.]*len(test_dataset)    
    TP3=0
    TN3=0
    FP3=0
    FN3=0
    #下面四个变量是两个参与分类器对同一个样本给出不同标签时用的变量
    TP30=0
    TN30=0
    FP30=0
    FN30=0
    uncertaincount=0
    
    #下面的变量用来统计不同投票情况下的TP,TN,FP,FN
    case1_TP3=0
    case1_TN3=0
    case1_FP3=0
    case1_FN3=0
    
    case2_TP3=0
    case2_TN3=0
    case2_FP3=0
    case2_FN3=0
    
    #下面几个变量是用标记或来统计不同投票情况下的TP,TN,FP,FN等情况
    case1_flag=0
    case2_flag=0
    '''
    #下面几个变量是有关概率加权的
    A10=0.8367     #Specificity of model one
    A11=0.9167      #Sensetivity of model one
    A20=0.8233      #Specificity of model two
    A21=0.86       #Sensetivity of model two
    
    W01=round(A10/(A10+A20),4) #Specificity weight of model one
    W02=round(A20/(A10+A20),4) #Specificity weight of model two
    W11=round(A11/(A11+A21),4) #Sensetivity weight of model one
    W12=round(A21/(A11+A21),4) #Sensetivity weight of model two
    print("W01="+str(W01)+"  W02="+str(W02)+"  W11="+str(W11)+"  W12="+str(W12))
    '''
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
            
        #数组y是真实值，和标签数组是一样的。只是标签在一次循环中是test_batchs个数，要把它们放到y中
        percentage1=torch.nn.functional.softmax(out1,dim=1)
        percentage2=torch.nn.functional.softmax(out2,dim=1)
        '''
        print('percentage1:')
        print(percentage1)
        print('percentage2:')
        print(percentage2) 
        '''
        if len(label)==test_batchs:
            endval=test_batchs
        else:
            endval=len(test_dataset)-row_i*test_batchs
                
        case1_flag=0
        case2_flag=0
        for i in range(endval):  #前面的批都够一个test_batchs，例如总数104，一个batchs是20，那么最后一个不够20个
            y[row_i*test_batchs+i]=label[i].item()
            per1=round(percentage1[i,1].item(),4) #mobilenetv1的正类概率
            per2=round(percentage2[i,1].item(),4) #mobilenetv2的正类概率
            per3=0.0     
            pred_label_tmp=0
            unclearflag=0
            
            if int(pred_label1)==int(pred_label2): #若两个模型的预测标签一样，则取二者预测概率的平均值
                per3=round((per1+per2)/2,4)
                pred_label_tmp=pred_label1[i].item()  
                unclearflag=1
                case1_flag=1
            else:                
                uncertaincount+=1
                case2_flag=1
                #print('count='+str(count))
                '''
                print('percentage1[i,0].item()='+str(round(percentage1[i,0].item(),4)))
                print('percentage2[i,0].item()='+str(round(percentage2[i,0].item(),4)))
                print('percentage1[i,1].item()='+str(round(percentage1[i,1].item(),4)))
                print('percentage2[i,1].item()='+str(round(percentage2[i,1].item(),4)))
                '''
                '''
                #下面两行代码是取两个模型对同一样本的阳性概率加权平均值及阴性概率加权平均值
                tmp1=round(percentage1[i,0].item()*W01+percentage2[i,0].item()*W02,4)
                tmp2=round(percentage1[i,1].item()*W11+percentage2[i,1].item()*W12,4)
                print('tmp1='+str(tmp1)+'  tmp2='+str(tmp2))
                #exit(0)
                '''
                 
                #下面两行代码是取两个模型对同一样本的阳性概率平均值及阴性概率平均值
                tmp1=round((percentage1[i,0].item()+percentage2[i,0].item())/2,4)
                tmp2=round((percentage1[i,1].item()+percentage2[i,1].item())/2,4)
                
                per3=tmp2
                if tmp1>tmp2:
                    pred_label_tmp=0
                else:
                    pred_label_tmp=1
                    
                '''
                print("real label:",label[i].item())
                print("V1 label:",pred_label1)
                print("V1 percentage:",percentage1)
                print("V2 label:",pred_label2)
                print("V2 percentage:",percentage2)
                '''
                '''
                tmp1=round(percentage1[i,pred_label1[i].item()].item(),4)
                tmp2=round(percentage2[i,pred_label2[i].item()].item(),4)            
                if tmp1>tmp2:  #若两个模型的预测标签不同，则取预测概率大的那个正类概率和对应的预测标签
                    per3=per1
                    pred_label_tmp=pred_label1[i].item()
                elif tmp1<tmp2:
                    per3=per2
                    pred_label_tmp=pred_label2[i].item()
                '''
            pred_per1[row_i*test_batchs+i]=round(per1,4)
            pred_per2[row_i*test_batchs+i]=round(per2,4)
            pred_per3[row_i*test_batchs+i]=round(per3,4)
            pred_label3=pred_label_tmp
        
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
                
            if case1_flag==1:  #两个分类器都给出相同的标签
                if (label[i].item()==1 and pred_label3==1):
                    case1_TP3+=1
                if (label[i].item()==0 and pred_label3==0):
                    case1_TN3+=1
                if (label[i].item()==1 and pred_label3==0):
                    case1_FN3+=1
                if (label[i].item()==0 and pred_label3==1):
                    case1_FP3+=1 

            if case2_flag==1:  #两个分类器中，两个给0,两个给1
                if (label[i].item()==1 and pred_label3==1):
                    case2_TP3+=1
                if (label[i].item()==0 and pred_label3==0):
                    case2_TN3+=1
                if (label[i].item()==1 and pred_label3==0):
                    case2_FN3+=1
                if (label[i].item()==0 and pred_label3==1):
                    case2_FP3+=1
                    
            '''
            if unclearflag==1:  #两个参与分类器给出的标签相同
                if (label[i].item()==1 and pred_label3==1):
                    TP3=TP3+1
                if (label[i].item()==0 and pred_label3==0):
                    TN3=TN3+1
                if (label[i].item()==1 and pred_label3==0):
                    FN3=FN3+1
                if (label[i].item()==0 and pred_label3==1):
                    FP3=FP3+1
                    
            if unclearflag==0:  #两个参与分类器给出的标签不相同
                if (label[i].item()==1 and pred_label3==1):
                    TP30=TP30+1
                if (label[i].item()==0 and pred_label3==0):
                    TN30=TN30+1
                if (label[i].item()==1 and pred_label3==0):
                    FN30=FN30+1
                if (label[i].item()==0 and pred_label3==1):
                    FP30=FP30+1 
            '''
                
        row_i=row_i+1
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
    
    TP3=case1_TP3+case2_TP3
    TN3=case1_TN3+case2_TN3
    FP3=case1_FP3+case2_FP3
    FN3=case1_FN3+case2_FN3
    
    sensitivity3=TP3/(TP3+FN3) #即TPR，可以通过求acc同时获得
    specificity3=TN3/(TN3+FP3) 
    G_Mean3=(sensitivity3*specificity3)**0.5  #开平方根
    error_rate3=(FP3+FN3)/(TP3+TN3+FP3+FN3)  #即FPR，可以通过求acc同时获得
    F_Measure3=2*TP3/(2*TP3+FP3+FN3)
    fpr3,tpr3,thresholds3=roc_curve(y,pred_per3,pos_label=1)
    roc_auc3 = round(auc(fpr3,tpr3),4)
    test_acc3=1-error_rate3 #当前epoch的测试准确率

    print("Evaluation index of "+modelname1+":")    
    print("test_acc is:",test_acc1)    
    print("sensitivity is:",sensitivity1) 
    print("specificity is:",specificity1)    
    print("error_rate is:",error_rate1)  
    print("auc is:",roc_auc1)    
    print("G_Mean is:",G_Mean1)    
    print("F_Measure is:",F_Measure1) 

    print("Evaluation index of "+modelname2+":")     
    print("test_acc is:",test_acc2)    
    print("sensitivity is:",sensitivity2) 
    print("specificity is:",specificity2)    
    print("error_rate is:",error_rate2)  
    print("auc is:",roc_auc2)    
    print("G_Mean is:",G_Mean2)    
    print("F_Measure is:",F_Measure2) 

    print("Evaluation index of "+modelname1+'&'+modelname2+":")     
    print("test_acc is:",test_acc3)    
    print("sensitivity is:",sensitivity3) 
    print("specificity is:",specificity3)    
    print("error_rate is:",error_rate3)  
    print("auc is:",roc_auc3)    
    print("G_Mean is:",G_Mean3)    
    print("F_Measure is:",F_Measure3)
    #print(modelname1+' Sensetivity:'+str(A11)+' Specificity:'+str(A10))
    #print(modelname2+' Sensetivity:'+str(A21)+' Specificity:'+str(A20))
    '''
    print('model: '+modelname1)
    print('the number samples of the right lable,TP='+str(TP1)+'   TN='+str(TN1))
    print('the number samples of the wrong lable,FP='+str(FP1)+'  FN='+str(FN1))
    
    print('model: '+modelname2)
    print('the number samples of the right lable,TP='+str(TP2)+'   TN='+str(TN2))
    print('the number samples of the wrong lable,FP='+str(FP2)+'  FN='+str(FN2))
    
    print('model: '+modelname1+'&'+modelname2)
    '''
    '''
    print('Two participating models give the same lable for a sample:')
    print('the number samples of the right lable,TP='+str(TP3)+'   TN='+str(TN3))
    print('the number samples of the wrong lable,FP='+str(FP3)+'  FN='+str(FN3))
    print('Two participating models give different lable for a sample:')    
    print('certain,TP3=',TP3,'   TN3=',TN3, '   FP3=',FP3,'  FN3=',FN3)
    print('uncertain,TP30=',TP30,'   TN30=',TN30, '   FP30=',FP30,'  FN30=',FN30)
    print("uncertaincount:",str(uncertaincount))
    '''
    print('case 1: TP3=',case1_TP3,'  TN3=',case1_TN3,'  FP3=',case1_FP3,'  FN3=',case1_FN3)
    print('case 2: TP3=',case2_TP3,'  TN3=',case2_TN3,'  FP3=',case2_FP3,'  FN3=',case2_FN3)





