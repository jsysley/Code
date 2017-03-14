# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:06:59 2017

@author: jsysley
"""
import numpy as np
import pandas as pd
import os
import sys
from script import * 

path = "F:\\code\\cta\\data\\factor"
list_file = os.listdir(path)
results = Read_All(path)
data_all = Deal_Factor(results,list_file)

####准备两份数据集验证
data1 = train1.copy()
data2 = train2.copy()

data1_with_y = Tag_Attach(data1,a,b) 
data2_with_y = Tag_Attach(data2,a,b) 
data1_ss,data2_ss = Data_PreDeal(data1_with_y,data2_with_y.drop('y',axis=1))

data1_ss_cut = data1_ss.drop('close',axis=1).copy()
data2_ss_cut = data2_ss.drop('close',axis=1).copy()

data1_ss_pca,data2_ss_pca = Get_Pca(data1_ss_cut,data2_ss_cut,-1)

data_temp1 = data1_ss_pca.copy()
data_temp2 = pd.concat([data2_ss_pca,data2_with_y.loc[:,'y']],axis=1)


