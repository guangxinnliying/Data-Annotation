# -*- coding: utf-8 -*-
#该程序使用论文中提出的整合方法，给出参与分类器和整合分类器的各项性能指标，如准确率，AUC等。
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
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3_Large
from models.vgg import VGG

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
recordfile='./bestmodels/resultrecord.txt'
file=open(recordfile,'w')

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
    count=0
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
        if len(label)==test_batchs:
            endval=test_batchs
        else:
            endval=len(test_dataset)-row_i*test_batchs
        
        for i in range(endval):  #前面的批都够一个test_batchs，例如总数104，一个batchs是20，那么最后一个不够20个
            y[row_i*test_batchs+i]=label[i].item()
            per1=round(percentage1[i,1].item(),4) #mobilenetv1的正类概率，这个数是用于计算AUC的，要求阳性（1，正类）的概率
            per2=round(percentage2[i,1].item(),4) #mobilenetv2的正类概率，这个数是用于计算AUC的，要求阳性（1，正类）的概率

            per3=0.0     
            pred_label_tmp=0
            
            if int(pred_label1)==int(pred_label2): #若两个模型的预测标签一样，则取二者预测概率的平均值
                per3=round((per1+per2)/2,4)
                pred_label_tmp=pred_label1[i].item()            
            else:                
                count+=1
                tmp1=round((percentage1[i,0].item()+percentage2[i,0].item())/2,4)
                tmp2=round((percentage1[i,1].item()+percentage2[i,1].item())/2,4)
                per3=tmp2 #这个数是用于计算AUC的，要求阳性（1，正类）的概率
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
            
            if (label[i].item()==1 and pred_label3==1):
                TP3=TP3+1
            if (label[i].item()==0 and pred_label3==0):
                TN3=TN3+1
            if (label[i].item()==1 and pred_label3==0):
                FN3=FN3+1
            if (label[i].item()==0 and pred_label3==1):
                FP3=FP3+1

                
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
    
    total_test_acc1+=test_acc1  
    total_sensitivity1+=sensitivity1
    total_specificity1+=specificity1
    total_error_rate1+=error_rate1
    total_roc_auc1+=roc_auc1
    total_G_Mean1+=G_Mean1
    total_F_Measure1+=F_Measure1
    
    total_test_acc2+=test_acc2  
    total_sensitivity2+=sensitivity2
    total_specificity2+=specificity2
    total_error_rate2+=error_rate2
    total_roc_auc2+=roc_auc2
    total_G_Mean2+=G_Mean2
    total_F_Measure2+=F_Measure2
    
    total_test_acc3+=test_acc3  
    total_sensitivity3+=sensitivity3
    total_specificity3+=specificity3
    total_error_rate3+=error_rate3
    total_roc_auc3+=roc_auc3
    total_G_Mean3+=G_Mean3
    total_F_Measure3+=F_Measure3
    
    file.write("current_fold is:"+str(current_time))
    file.write('Evaluation index of '+modelname1+'\n')
    file.write("test_acc is:"+str(round(test_acc1,4))+'\n')    
    file.write("sensitivity is:"+str(round(sensitivity1,4))+'\n')
    file.write("specificity is:"+str(round(specificity1,4))+'\n')
    file.write("error_rate is:"+str(round(error_rate1,4))+'\n')
    file.write("auc is:"+str(round(roc_auc1,4))+'\n')
    file.write("G_Mean is:"+str(round(G_Mean1,4))+'\n')
    file.write("F_Measure is:"+str(round(F_Measure1,4))+'\n')
    file.write('\n')
    file.write('Evaluation index of '+modelname2+'\n')
    file.write("test_acc is:"+str(round(test_acc2,4))+'\n')    
    file.write("sensitivity is:"+str(round(sensitivity2,4))+'\n')
    file.write("specificity is:"+str(round(specificity2,4))+'\n')
    file.write("error_rate is:"+str(round(error_rate2,4))+'\n')
    file.write("auc is:"+str(round(roc_auc2,4))+'\n')
    file.write("G_Mean is:"+str(round(G_Mean2,4))+'\n')
    file.write("F_Measure is:"+str(round(F_Measure2,4))+'\n')
    file.write('\n')
    file.write('Evaluation index of '+modelname1+'&'+modelname2+'\n')
    file.write("test_acc is:"+str(round(test_acc3,4))+'\n')    
    file.write("sensitivity is:"+str(round(sensitivity3,4))+'\n')
    file.write("specificity is:"+str(round(specificity3,4))+'\n')
    file.write("error_rate is:"+str(round(error_rate3,4))+'\n')
    file.write("auc is:"+str(round(roc_auc3,4))+'\n')
    file.write("G_Mean is:"+str(round(G_Mean3,4))+'\n')
    file.write("F_Measure is:"+str(round(F_Measure3,4))+'\n')
    file.write('\n')
    file.write('\n')

    
    print('Evaluation index of '+modelname1)    
    print("test_acc is:",test_acc1)    
    print("sensitivity is:",sensitivity1) 
    print("specificity is:",specificity1)    
    print("error_rate is:",error_rate1)  
    print("auc is:",roc_auc1)    
    print("G_Mean is:",G_Mean1)    
    print("F_Measure is:",F_Measure1) 

    print('Evaluation index of '+modelname2)    
    print("test_acc is:",test_acc2)    
    print("sensitivity is:",sensitivity2) 
    print("specificity is:",specificity2)    
    print("error_rate is:",error_rate2)  
    print("auc is:",roc_auc2)    
    print("G_Mean is:",G_Mean2)    
    print("F_Measure is:",F_Measure2) 

    print('Evaluation index of '+modelname1+'&'+modelname2)    
    print("test_acc is:",test_acc3)    
    print("sensitivity is:",sensitivity3) 
    print("specificity is:",specificity3)    
    print("error_rate is:",error_rate3)  
    print("auc is:",roc_auc3)    
    print("G_Mean is:",G_Mean3)    
    print("F_Measure is:",F_Measure3)  

    print("count:",count)

if total_time>1:
    file.write('Avg of evaluation index of '+modelname1+'\n')
    file.write("test_acc is:"+str(round(total_test_acc1/total_time,4))+'\n')    
    file.write("sensitivity is:"+str(round(total_sensitivity1/total_time,4))+'\n')
    file.write("specificity is:"+str(round(total_specificity1/total_time,4))+'\n')
    file.write("error_rate is:"+str(round(total_error_rate1/total_time,4))+'\n')
    file.write("auc is:"+str(round(total_roc_auc1/total_time,4))+'\n')
    file.write("G_Mean is:"+str(round(total_G_Mean1/total_time,4))+'\n')
    file.write("F_Measure is:"+str(round(total_F_Measure1/total_time,4))+'\n')
    file.write('\n')
    file.write('Avg of evaluation index of '+modelname2+'\n')
    file.write("test_acc is:"+str(round(total_test_acc2/total_time,4))+'\n')    
    file.write("sensitivity is:"+str(round(total_sensitivity2/total_time,4))+'\n')
    file.write("specificity is:"+str(round(total_specificity2/total_time,4))+'\n')
    file.write("error_rate is:"+str(round(total_error_rate2/total_time,4))+'\n')
    file.write("auc is:"+str(round(total_roc_auc2/total_time,4))+'\n')
    file.write("G_Mean is:"+str(round(total_G_Mean2/total_time,4))+'\n')
    file.write("F_Measure is:"+str(round(total_F_Measure2/total_time,4))+'\n')
    file.write('\n')
    file.write('Avg of evaluation index of '+modelname1+'&'+modelname2+'\n')
    file.write("test_acc is:"+str(round(total_test_acc3/total_time,4))+'\n')    
    file.write("sensitivity is:"+str(round(total_sensitivity3/total_time,4))+'\n')
    file.write("specificity is:"+str(round(total_specificity3/total_time,4))+'\n')
    file.write("error_rate is:"+str(round(total_error_rate3/total_time,4))+'\n')
    file.write("auc is:"+str(round(total_roc_auc3/total_time,4))+'\n')
    file.write("G_Mean is:"+str(round(total_G_Mean3/total_time,4))+'\n')
    file.write("F_Measure is:"+str(round(total_F_Measure3/total_time,4))+'\n')
    file.write('\n')


    print('Avg of evaluation index of '+modelname1)    
    print("test_acc is:",round(total_test_acc1/total_time,4))    
    print("sensitivity is:",round(total_sensitivity1/total_time,4)) 
    print("specificity is:",round(total_specificity1/total_time,4))    
    print("error_rate is:",round(total_error_rate1/total_time,4))  
    print("auc is:",round(total_roc_auc1/total_time,4))    
    print("G_Mean is:",round(total_G_Mean1/total_time,4))    
    print("F_Measure is:",round(total_F_Measure1/total_time,4))

    print('Avg of evaluation index of '+modelname2)    
    print("test_acc is:",round(total_test_acc2/total_time,4))    
    print("sensitivity is:",round(total_sensitivity2/total_time,4)) 
    print("specificity is:",round(total_specificity2/total_time,4))    
    print("error_rate is:",round(total_error_rate2/total_time,4))  
    print("auc is:",round(total_roc_auc2/total_time,4))    
    print("G_Mean is:",round(total_G_Mean2/total_time,4))    
    print("F_Measure is:",round(total_F_Measure2/total_time,4))

    print('Avg of evaluation index of '+modelname1+'&'+modelname2)    
    print("test_acc is:",round(total_test_acc3/total_time,4))    
    print("sensitivity is:",round(total_sensitivity3/total_time,4)) 
    print("specificity is:",round(total_specificity3/total_time,4))    
    print("error_rate is:",round(total_error_rate3/total_time,4))  
    print("auc is:",round(total_roc_auc3/total_time,4))    
    print("G_Mean is:",round(total_G_Mean3/total_time,4))    
    print("F_Measure is:",round(total_F_Measure3/total_time,4))
file.close()   
'''
rocdatafile='./rocdatafile/MobileNet/mobilev1_roc.txt'
f=open(rocdatafile,'w')
f.write(str(fpr1)+'\n')
f.write(str(tpr1)+'\n')
f.close()

rocdatafile='./rocdatafile/MobileNet/mobilev2_roc.txt'
f=open(rocdatafile,'w')
f.write(str(fpr2)+'\n')
f.write(str(tpr2)+'\n')
f.close()

rocdatafile='./rocdatafile/MobileNet/mobilev1andv2_roc.txt'
f=open(rocdatafile,'w')
f.write(str(fpr3)+'\n')
f.write(str(tpr3)+'\n')
f.close()
'''




