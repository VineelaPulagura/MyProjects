# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23  2021
This is the program we used to plot filter result and show on the paper
@author: Shen Po-Heng
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Preprocess_tool():
    @staticmethod    
    def seperate_toxyz(data):
        x = np.array(data_acc[0])
        y = np.array(data_acc[1])
        z = np.array(data_acc[2])
        return x,y,z
    @staticmethod    
    def Filter_toBodyGravity(Signal):
        LowFilter = signal.butter(3, 0.3, 'low', fs=50, output='sos') #low-pass filter(cutoff frequency=0.3Hz)
        Body_part = signal.sosfilt(LowFilter, Signal)
        Gra_part=Signal-Body_part
        return Body_part,Gra_part


data_acc = pd.read_csv('HAPT Data Set/RawData/ACC/acc_exp01_user01.txt',sep='\s+',header=None) #read data from the dataset
data_acc = pd.DataFrame(data_acc) #..


Tool=Preprocess_tool()
#for 
Acc_x,Acc_y,Acc_z=Tool.seperate_toxyz(data_acc)

fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_title('Before filter(Acc_x)')
ax1.set_xlabel('Time [seconds]')
ax1.axis([0, len(Acc_x), -2, 2])
ax1.plot(Acc_x)

#Butter Filter
tBodyAcc_x,tGravityAcc_x=Tool.Filter_toBodyGravity(Acc_x)


ax2.set_title('After low-pass filter(cutoff frequency=0.3Hz)\n Body part')
ax2.axis([0, len(tBodyAcc_x), -2, 2])
ax2.set_xlabel('Time [seconds]')
ax2.plot(tBodyAcc_x)
plt.tight_layout()

ax3.set_title('Gravity Part')
ax3.axis([0, len(tGravityAcc_x), -2, 2])
ax3.set_xlabel('Time [seconds]')
ax3.plot(tGravityAcc_x)
plt.tight_layout()
plt.show()

