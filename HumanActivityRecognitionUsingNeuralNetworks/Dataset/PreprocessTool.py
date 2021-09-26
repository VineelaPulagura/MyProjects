# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21  2021
This is the tool function for preprocess
@author: Shen Po-Heng
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import iqr
from scipy import signal


class Create_Table:
    #combine all extraction data into one input
    def Create_Table(DataPath,savePath):    
        Table=pd.DataFrame()#Declare a table for combine extracted data into one table
        print("\ncombine all users' extracted data....")
        for file in os.listdir(DataPath):
            #Read data
            print("Loading... %s"% str(file))
            data = pd.DataFrame(pd.read_csv(DataPath+str(file), index_col=0))
            Table=pd.concat([Table,data], ignore_index=False, sort=False)
            # Table.append(data) #another way to combine    
        Table.rename(columns={str(len(Table.columns)-1):"label"}, inplace=True)
        # print(str(len(Table.columns)-2))
        Table.to_csv('%sexp_RawExtractionResult_Filtered_ALL_Gyro_nofilter.csv' % savePath,index=False)
        
class extraction_tool:
    #extract value from a window
    def extract_values(self,rowData,data):
        rowData.append(np.mean(data))# get Mean Value in a window
        rowData.append(abs(np.median(data)))#Median Value
        rowData.append(np.var(data))#Variance
        rowData.append(np.std(data))#deviation Value
        rowData.append(iqr(data))#IQR value
        rowData.append(np.max(data))#Max Value
        rowData.append(np.min(data))#Min Value
    #savedata
    def save_data(self,file,Data,savepath):
        TableData=pd.DataFrame(Data)
        TableData.to_csv('%sexp_RawExtractionResult_%s.csv' % (savepath ,str(file[14:16])))
        print('Save exp_RawExtractionResult_%s.csv to %s' % (str(file[14:16]),savepath))
        
class Preprocess_tool():
    @staticmethod    
    def seperate_toxyz(data):
        x = np.array(data[0])
        y = np.array(data[1])
        z = np.array(data[2])
        return x,y,z
    #this is use lowpass filter to separate gravity and body part
    @staticmethod    
    def Filter_toBodyGravity(Signal):
        LowFilter = signal.butter(3, 0.3, 'low', fs=50, output='sos') #low-pass filter(cutoff frequency=0.3Hz)
        Body_part = signal.sosfilt(LowFilter, Signal)
        Gra_part=Signal-Body_part
        return Body_part,Gra_part


class ThreeDim_Preprocess(Preprocess_tool):#especially for XYZ data
    #using the Filter_toBodyGravity function in 3 diffetent direction
    def Acc_toBodyGravity(self,Signal):#call Filter_toBodyGravity() 3 times
        Body,Gravity=np.zeros((3,len(Signal[0]))),np.zeros((3,len(Signal[0])))
        for i in range(3):
            Body[i],Gravity[i]=super().Filter_toBodyGravity(Signal[i])
        return Body,Gravity