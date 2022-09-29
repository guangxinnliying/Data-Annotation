# -*- coding: UTF-8 -*-
#python 一个折线图绘制多个曲线
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
#这里导入你自己的数据
x_axis = [1,2,3,4,5,6,7,8,9,10] #x

##The following code draws the accuracy comparison chart of mobilenetv2
y_axis_mobilenetv2_acc1= [0.8617,0.875,0.8517,0.8667,0.86,0.8483,0.8783,0.8433,0.85,0.86
] # Accuracy of mobilev1 with one-phase transfer
y_axis_mobilenetv2_acc2= [0.88,0.89,0.8867,0.8867,0.89,0.885,0.8883,0.8867,0.8817,0.8883
] # Accuracy of mobilev1 with two-phase transfer

#开始画图

#plt.title('Accuracy comparison of MobileNetV2')
plt.plot(x_axis,y_axis_mobilenetv2_acc1, 'g*--', label='The one-phase transfer learning')
plt.plot(x_axis, y_axis_mobilenetv2_acc2, 'ro--',label='The two-phase transfer learning')

plt.legend() # 显示图例
 
plt.xlabel('Number of times')
plt.ylabel('Accuracy')
plt.show()

'''
##The following code draws the AUC comparison chart of mobilenetv2
y_axis_mobilenetv2_auc1= [0.9218,0.9338,0.9022,0.9319,0.9319,0.9014,0.9347,0.9048,0.9202,0.9267] # Accuracy of mobilev1 with one-phase transfer
y_axis_mobilenetv2_auc2= [0.9341,0.9384,0.9332,0.9326,0.9357,0.9373,0.9421,0.9338,0.9308,0.9353] # Accuracy of mobilev1 with two-phase transfer

#plt.title('AUC comparison of MobileNetV2')
plt.plot(x_axis,y_axis_mobilenetv2_auc1, 'g*--', label='The one-phase transfer learning')
plt.plot(x_axis, y_axis_mobilenetv2_auc2, 'ro--',label='The two-phase transfer learning')

plt.legend() # 显示图例
 
plt.xlabel('Number of times')
plt.ylabel('AUC')
plt.show()
##the AUC comparison chart of mobilenetv2
'''
'''
##The following code draws the accuracy comparison chart of mobilenetv3_large
y_axis_mobilenetv3_acc1= [0.5117,0.54,0.555,0.5133,0.5383,0.5067,0.635,0.5133,0.6367,0.5817] # Accuracy of mobilenetv3_large with one-phase transfer
y_axis_mobilenetv3_acc2= [0.8517,0.8417,0.8567,0.85,0.8767,0.85,0.8467,0.86,0.865,0.865] # Accuracy of mobilenetv3_large with two-phase transfer


#plt.title('Accuracy comparison of MobileNetV3_Large')
plt.plot(x_axis,y_axis_mobilenetv3_acc1, 'g*--', label='The one-phase transfer learning')
plt.plot(x_axis,y_axis_mobilenetv3_acc2, 'ro--',label='The two-phase transfer learning')

plt.legend() # 显示图例
 
plt.xlabel('Number of times')
plt.ylabel('Accuracy')
plt.show()
##the accuracy comparison chart of mobilenetv3_large
'''
'''
##The following code draws the AUC comparison chart of mobilenetv3_large
y_axis_mobilenetv3_auc1= [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] # AUC of mobilenetv3_large with one-phase transfer
y_axis_mobilenetv3_auc2= [0.9317,0.9303,0.9222,0.9198,0.948,0.9104,0.9286,0.9289,0.9328,0.9343] # AUC of mobilenetv3_large with two-phase transfer

#plt.title('AUC comparison of MobileNetV3_Large')
plt.plot(x_axis,y_axis_mobilenetv3_auc1, 'g*--', label='The one-phase transfer learning')
plt.plot(x_axis,y_axis_mobilenetv3_auc2, 'ro--',label='The two-phase transfer learning')

plt.legend() # 显示图例
 
plt.xlabel('Number of times')
plt.ylabel('AUC')
plt.show()
##the AUC comparison chart of mobilenetv3_large
'''