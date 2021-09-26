# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15  2021
This program is only to use raw data with filter for accleration to do segmentation and extraction

Caution: when run the code, please check the datapath and set up the rawdata file like:
    Accleration data in "ACC" file folder, Angular data(from gyroscope) in "Gyro" file folder
@author: Shen Po-Heng
"""
import pandas as pd
import numpy as np
import os
from PreprocessTool import Create_Table,extraction_tool,ThreeDim_Preprocess



#setup the path
savePath = "Table/ExtractionResult/"
DataPath_ACC= "RawData/ACC/"
LabeldataPath="RawData/labels.txt"
label=np.array(pd.read_csv(LabeldataPath,sep='\s+',header=None))
i=0
TableData=[]#record one user data(1st->intitialization)

#declare class
ex=extraction_tool()
Tool=ThreeDim_Preprocess() 
BuildTable=Create_Table()

#adjust length and Overlap from here
Length_SlidingWindow=128
Overlap=64 #Overlap/Length_SlidingWindow=Percentage  
'''
Overlap 50% is up to the length of window. For an example, if our window length is at 50 reading per window.
The overlap 50 % is to set up overlap at 25, if 33.33%, to set up overlap at around 17.....and so on
''' 

#Segmentation and Extraction
for file in os.listdir(DataPath_ACC):
    #Read data
    print("Loading...experiment%s user%s" % (str(file[7:9]),str(file[14:16])))
    data_acc = pd.read_csv(DataPath_ACC+str(file),sep='\s+',header=None)
    data_gyra = pd.read_csv(DataPath_ACC.replace("ACC","Gyro")+str(file).replace("acc","gyro"),sep='\s+',header=None)
    data_acc = np.transpose(np.array(data_acc))#data_acc[0] is x;data_acc[1] is y;data_acc[2] is z
    data_gyra = np.transpose(np.array(data_gyra)) 
    tBodyAcc_XYZ,tGravityAcc_XYZ=Tool.Acc_toBodyGravity(data_acc)#filter
    #Combine
    data=np.append(tBodyAcc_XYZ,tGravityAcc_XYZ, axis=0)
    data=np.append(data,data_gyra, axis=0)
    while 1:
        #sliding window->it would run one activity once, a window is 128 data and their overlap is 50%
        for k in range(label[i][3],label[i][4],Overlap):#run one activity
            rowData=[]
            if k+Length_SlidingWindow <= label[i][4]:
                for j in range(0,len(data),1):#extract 128 data in 33 features (len(data)=33
                    ex.extract_values(rowData,data[j][k:k+Length_SlidingWindow])
            else:
                break;                
            rowData.append(label[i][2])#record one activity
            TableData.append(rowData)#collect all activity data into one table for one user
        if i==len(label)-1 :
            ex.save_data(str(file),TableData,savePath)
            break
        else:
            if  label[i+1][1]!=label[i][1]: #next user is same or not; different->save
                ex.save_data(str(file),TableData,savePath)
                TableData=[]#record one user data
                i+=1#to another activity
                break
            elif label[i+1][0] !=label[i][0]: # user still same person but experiment is different
                i+=1#to another activity
                break
            else: #same person same experiment
                i+=1#to another activity
#Build Table
Datapath=savePath
SavePath="Table/"
Create_Table.Create_Table(Datapath,SavePath)
#test and see the result 
testDF = pd.DataFrame(pd.read_csv('%sexp_RawExtractionResult_Filtered_ALL_Gyro_nofilter.csv' % "Table/"))
print(testDF)   



