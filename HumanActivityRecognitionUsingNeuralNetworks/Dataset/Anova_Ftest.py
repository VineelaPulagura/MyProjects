# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15  2021
This is to accesess the original data and do Anove Ftest for select the relevant features
We can adjust threshold to select features
The higher the score it get in Anove Ftest, the more relevant it is
@author: Shen Po-Heng
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif #Reference library:https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
import matplotlib.pyplot as plt
  
dropNum=[] #for deop features
RelevanceNum=[] #for relevant features
threshold=3000 # setup the value of threshold to filter features

# load the dataset
df = pd.read_csv('Table/exp_RawExtractionResult_Filtered_ALL_Gyro_nofilter.csv',index_col=[0])
data_arr=np.array(df)
print(df)

#ANOVA Ftest
Y=df['label']
X=df.drop('label',1)
# configure to select all features
fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship: value of every features would compared with label value
fs.fit(X, Y)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    if fs.scores_[i] < threshold :
        dropNum.append(i)
    else:
        RelevanceNum.append(i)
        
# plot the scores
fig = plt.figure()
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Anova-ftest")
fig.savefig("Features.png",dpi=300)

# Show results
print("Feature:", dropNum)
#drop those weak relation features
for i in range(len(dropNum)-1):
    df=df.drop([str(dropNum[i])],1)
df.to_csv('Table/FSexp_Result.csv')

#test and see the result 
testDF = pd.DataFrame(pd.read_csv('Table/FSexp_Result.csv'))
print(testDF) 