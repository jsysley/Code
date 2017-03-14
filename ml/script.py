 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:38:20 2017

@author: jsysley
"""
#####
#####分块读取文件，读取单个文件
#path为factor文件夹路径，返回一个函数
def Read_Facotr(path):
    def Read_Factor_in(file_name):
        file_path = path+"/"+file_name
        reader = pd.read_csv(file_path,iterator=True)
        #批量分块读取数据
        try:
            df = reader.get_chunk(800)
        except StopIteration:
            print "Iteration is stopped."
        return df
    return Read_Factor_in
    
#####
#####并行读入文件，读入所有文件
#返回一个list
def Read_All(path):
    Read = Read_Facotr(path)
    #批量读入数据
    pool = ThreadPool(3)
    results = pool.map(Read, list_file)#44个指标
    pool.close()
    pool.join()
    return results

#####
#####将读入的文件写成一张表
#results为Read_All返回的函数结果，list_file为文件列表
def Deal_Factor(results,list_file):
    names = map(lambda x:x.split(".")[0],list_file)
    def Concat(data1,data2):
        return(pd.concat([data1,data2]))
    data_res = reduce(Concat,results)
    data_res.index = names
    data_res = data_res.T
    return data_res
    
###
###列出所有csv文件
def List_File_Csv(path):
    os.chdir(path)
    names = glob.glob("*.csv")
    return names

#####
#####数据贴y标签，回撤a,收益b
#对当前一个样本判断y，price为所有的close
#a为回撤，b为收益，index为判断到的第index个close，返回一个函数
def Get_A_Tag(price,a,b):
    def Get_Get(index):
        data_temp = price[index:]#拿出index及以后的价格
        price_now = price[index]#当前价格
        #计算出沿途的所有收益率
        along_return = (data_temp-price_now)/price_now
        down = along_return[along_return<a].index#拿出大于回撤的index
        down_index = map(lambda x:along_return.index.get_loc(x),down)#得到所有回测大于a的位置号
        up = along_return[along_return>=b].index
        up_index = map(lambda x:along_return.index.get_loc(x),up)#得到所有收益大于b的位置
        #两者为空的情况
        if (len(down_index)+len(up_index))==0:
            return "NA"
        #其一为空的情况
        elif len(down_index)==0:
            return "1"
        elif len(up_index)==0:
            return "0"
        #两者不为空的情况,判断首次位置
        if down_index[0] < up_index[0]:
            return "0"
        elif down_index[0] > up_index[0]:
            return "1"
    return Get_Get

#####      
#####给训练集train1贴y,回撤a（负数），收益b
#train是整个训练集，index是第几个样本
#返回有y的训练集（样本数可能与传入的样本数不一致，因为有的样本y贴不上）
def Tag_Attach(train,a,b):
    train1 = train.copy()
    price = train1.loc[:,'close']
    get_temp = Get_A_Tag(price,a,b)
    all_y = map(lambda x:get_temp(x),np.arange(len(price)))
    all_y_ser = pd.Series(all_y,index = price.index)
    all_y_ser.value_counts()
    #贴y标签
    train1.loc[all_y_ser=='1','y'] = 1
    train1.loc[all_y_ser=='0','y'] = 0
    train1.loc[all_y_ser=='NA','y'] = np.nan
    #取出有y的训练集
    train1_sub = train1[train1.loc[:,'y'].notnull()]
    return train1_sub

#####
#####标准化
#返回一个标准化后的数据集
def Special_Scale(data_A):
    dataA = data_A.copy()
    name = ["ChandelierExit_Long",
              "ChandelierExit_Short",
              "IchimokuClouds_BaseLine",
              "IchimokuClouds_ConversionLine",
              "KAMA_I",
              "MassIndex_I",
              "MovingAverageEnvelopes_LowerEnvelope",
              "MovingAverageEnvelopes_UpperEnvelope",
              "MovingAverages_ExponentialMovingAverage",
              "MovingAverages_SimpleMovingAverage",
              "Bollinger_BandWidth",
              "Detrended_Price_Oscillator",
              "EMA",
              "Know_Sure_Thing",
              "Parabolic_SAR"]
    scale_columns = dataA.columns[map(lambda x: x in name,dataA.columns.values)]
    dataA.loc[:,scale_columns] = dataA.loc[:,scale_columns].divide(dataA.loc[:,'close'],axis=0)
    return dataA

#####          
#####标准化
#返回标准化后的数据集
def Scale(data_A,data_A2):
    dataA = data_A.copy()
    dataA2 = data_A2.copy()
    ss = StandardScaler()
    
    dataA_x = dataA.drop(['close','y'],axis=1)
    dataA_y = dataA.loc[:,['close','y']]

    ss.fit(dataA_x)
    #dataA处理
    dataA_x_ss = ss.transform(dataA_x)
    dataA_x_ss = pd.DataFrame(dataA_x_ss,columns=dataA_x.columns,index=dataA_x.index)
    dataA_ss = pd.concat([dataA_x_ss,dataA_y],axis=1)
    #dataA2处理
    dataA2_x = dataA2.drop('close',axis=1)
    dataA2_close = dataA2.loc[:,'close']
    dataA2_x_ss = ss.transform(dataA2_x)
    dataA2_x_ss = pd.DataFrame(dataA2_x_ss,columns=dataA2_x.columns,index=dataA2_x.index)
    dataA2_ss = pd.concat([dataA2_x_ss,dataA2_close],axis=1)
    return dataA_ss,dataA2_ss

#####
#####数据预处理
#处理na值，然后标准化两个数据集
def Data_PreDeal(data_A,data_A2):
    dataA = data_A.copy()
    dataA2 = data_A2.copy()
    #缺失值处理
    dataA.fillna(0,inplace=True)
    dataA2.fillna(0,inplace=True)
    #除去价格因素
    dataA = Special_Scale(dataA)
    dataA2 = Special_Scale(dataA2)
    #数据标准化
    dataA_ss,dataA2_ss = Scale(dataA,dataA2)
    return dataA_ss,dataA2_ss
 
#####
#####主成分函数
#返回主成分转化后的两个数据集,n_component为-1时表示返回所有主成分
#传入的数据集dataA比dataA多一个’y’
def Get_Pca(dataA,dataA2,n_component):
    dataA_x = dataA.drop('y',axis=1).copy()
    dataA_y = dataA.loc[:,'y'].copy()
    
    dataA2_x = dataA2.copy()

    if n_component == -1:
        n_component = len(dataA_x.columns)
    pca = PCA(n_components=n_component)
    pca.fit(dataA_x)    
    #print(pca.explained_variance_ratio_) 
    #对dataA处理
    dataA_x_pca = pca.transform(dataA_x)
    col_name = map(lambda x: 'x'+str(x),np.arange(len(dataA_x.columns)))
    dataA_x_pca = pd.DataFrame(dataA_x_pca,index=dataA_x.index,columns=col_name)
    dataA_pca = pd.concat([dataA_x_pca,dataA_y],axis=1)
    #对dataA2处理
    dataA2_x_pca = pca.transform(dataA2_x)
    dataA2_x_pca = pd.DataFrame(dataA2_x_pca,index=dataA2_x.index,columns=col_name)
    dataA2_pca = dataA2_x_pca.copy()
    return(dataA_pca,dataA2_pca)


#####
#####随机森林变量选择
def Choose_RF(dataA,max_num):
    data_choose = dataA.copy()
    rf = RandomForestClassifier(n_jobs=-1,random_state=1,criterion='gini',oob_score=True)
    ###参数优化设置
    n_f = np.floor(np.linspace(1,(np.sqrt(data_choose.shape[1])-1),3)).astype('int64')
    n_e = np.floor(np.linspace(150,np.sqrt(data_choose.shape[0])*10,5)).astype('int64')
    n_e = list(n_e)
    n_e.extend([5,15,40,60,80])
    parameters = {'n_estimators':n_e,'max_features':n_f}
    ###交叉验证参数优化
    grid = GridSearchCV(rf,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=2,n_jobs=-1)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    feature_importance = grid.best_estimator_.feature_importances_
    #拿出重要性大于0的index
    important_idx = np.where(feature_importance > 0)[0]
    sorted_idx = list(np.argsort(feature_importance[important_idx])[::-1])
    num = min(max_num,len(sorted_idx))
    variables_choose = list(data_choose.columns[important_idx][sorted_idx][np.arange(num)])
    variables_score = feature_importance[important_idx][sorted_idx][np.arange(num)]
    return variables_choose,variables_score

#####
#####logistic变量选择
def Choose_LR(dataA,max_num):
    data_choose = dataA.copy()
    lr = LogisticRegression(penalty='l1',random_state=1,solver='liblinear')
    ###参数优化设置
    C = np.linspace(0,100,50)
    parameters = {'C':C}
    ###交叉验证参数优化
    grid = GridSearchCV(lr,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=2,n_jobs=-1)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    feature_importance = grid.best_estimator_.coef_[0]
    #拿出重要性大于0的index
    important_idx = np.where(feature_importance > 0)[0]
    sorted_idx = list(np.argsort(feature_importance[important_idx])[::-1])
    num = min(max_num,len(sorted_idx))
    variables_choose = list(data_choose.columns[important_idx][sorted_idx][np.arange(num)])
    variables_score = feature_importance[important_idx][sorted_idx][np.arange(num)]
    return variables_choose,variables_score

#####
#####svm_rfe变量选择
def Choose_SvmRfe(dataA,max_num):
    data_choose = dataA.copy()
    clf = SVC(random_state=1,verbose=True,kernel='linear')
    ###参数优化设置
    #C = np.array([0.001,0.01,0.1,1,10])
    #gamma = np.array([0.001,0.01,0.1,1,10])
    #parameters = {'C':C,'gamma':gamma}
    ###交叉验证参数优化
    #grid = GridSearchCV(clf,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=2,n_jobs=-1)
    #grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    #clf_best = grid.best_estimator_
    #selector =  RFECV(clf_best,step=1,cv=10,verbose=2)
    selector = RFE(clf_best,n_features_to_select=max_num,step=1)
    selector.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    variables_choose = list(data_choose.columns[selector.get_support()])
    variables_score = selector.ranking_[selector.get_support()]
    return variables_choose,variables_score
    
#####
####变量选择总函数
def Choose_Var(dataA,max_var):
    data_choose = dataA.copy()
    rn_var,rn_score = Choose_RF(data_choose,15)
    lr_var,lr_score = Choose_LR(data_choose,15)
    rfe_var,rfe_score = Choose_SvmRfe(data_choose,15)
    
    ss_rn = StandardScaler()
    ss_lr = StandardScaler()
    ss_rfe = StandardScaler()
    ss_rn.fit(rn_score)
    ss_lr.fit(lr_score)
    ss_rfe.fit(rfe_score)

    rn_score_ss = ss_rn.transform(rn_score)
    lr_score_ss = ss_lr.transform(lr_score)
    rfe_score_ss = ss_rfe.transform(rfe_score)
    
    rn_pack = zip(rn_var,rn_score_ss)
    lr_pack = zip(lr_var,lr_score_ss)
    rfe_pack = zip(rfe_var,rfe_score_ss)
    #整合统计结果
    var_score = dict(rn_pack)
    for i in lr_pack:
        var_score[i[0]] = var_score[i[0]].get(i[0],0) + i[1]
    for i in rn_pack:
        var_score[i[0]] = var_score[i[0]].get(i[0],0) + i[1]
    #排序
    sortedClass = sorted(var_score.iteritems(),key=operator.itemgetter(1),reverse=True)

    var_all_sortded = sorted(var_score.keys())
    num = min(len(var_all_sortded),max_var)
    choose_var = var_all_sortded[0:num]
    return choose_var

#####
#####核心模型GBDT
def GDBT(dataA):
    data_choose = dataA.copy()
    gdbt = GradientBoostingClassifier(random_state=1)
    #参数优化设置
    loss = ['deviance','exponential']
    learning_rate = [0.001,0.01,0.1,0.5,1]
    n_estimators = [10,30,50,70,100,200]
    subsample = [0.5,0.8,1]
    parameters = {'loss':loss,'learning_rate':learning_rate,'n_estimators':n_estimators,'subsample':subsample}
    ###交叉验证参数优化
    grid = GridSearchCV(gdbt,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=1,n_jobs= 2)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    model = grid.best_estimator_
    return model

#####
#####Extremely randomized tree
def Extra_RF(dataA):
    data_choose = dataA.copy()
    exrf = ExtraTreesClassifier(criterion='gini',bootstrap=True,random_state=1)
    #参数优化设置
    n_estimators = [10,30,50,70,100,200]
    max_features = [0.5,0.8,1]
    parameters = {'n_estimators':n_estimators,'max_features':max_features}
    ###交叉验证参数优化
    grid = GridSearchCV(exrf,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=2,n_jobs= 2)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    model = grid.best_estimator_
    return model

#####
#####SVM   
def Svm(dataA):
    data_choose = dataA.copy()
    svm = SVC(kernel='rbf',random_state=1)
    #参数优化设置
    C = [0.001,0.01,0.1,1,10]
    gamma = [0.001,0.01,0.1,1,10]
    parameters = {'C':C,'gamma':gamma}
    ###交叉验证参数优化
    grid = GridSearchCV(svm,param_grid=parameters,cv=5,scoring='precision',error_score=0,verbose=2,n_jobs= 2)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    model = grid.best_estimator_
    return model

#####
#####xgb
def Xgb(dataA):
    data_choose = dataA.copy()
    paras = {'silent':1,'colsample_bytree':0.75,'subsample':0.8,'eval_metric':'auc','seed':1, 'objective':'binary:logistic' }
    xgb = XGBClassifier(paras)
    #参数优化设置
    gamma = [0.01,0.1,1]
    n_estimators = [10,50,80,100,200]
    max_depth=[2,3,5]
    learning_rate = [0.001,0.01,0.1,1]
    reg_alpha = [0.001,0.01,1]
    reg_lambda = [0.001,0.01,1]
    parameters = {'n_estimators':n_estimators,'max_depth':max_depth,
    'learning_rate':learning_rate,'gamma':gamma,'reg_alpha':reg_alpha,'reg_lambda':reg_lambda}
    ###交叉验证参数优化
    grid = GridSearchCV(xgb,param_grid=parameters,cv=3,scoring='precision',error_score=0,verbose=1,n_jobs= 2)
    grid.fit(data_choose.drop('y',axis=1),data_choose.loc[:,'y'])
    model = grid.best_estimator_
    return model

#####
#####第一层
def Model_First(dataA):
    data_choose = dataA.copy()
    split = np.floor(data_choose.shape[0]*0.65).astype('int64')
    data_choose1 = data_choose.iloc[:split,:]
    data_choose2 = data_choose.iloc[split:,:]
    
    gbdt_model = GDBT(data_choose1)
    exrf_model = Extra_RF(data_choose1)
    svm_model = Svm(data_choose1)
    xgb_model = Xgb(data_choose1)
    
    
    gbdt_score = gdbt

    
def Deal_Proba(dataA2,model,threshold):
    pro = model.predict(dataA2)
    label = pd.Series(0*len(pro),index = dataA2.index)
    label[pro>threshold]=1
    return pro,label

#####
#####主函数
def Main(train1,train2,test,a,b,max_var,threshold):
    #贴标签
    train_with_y = Tag_Attach(train1,a,b)
    #返回标准化后的两个训练集
    train1_ss,train2_ss = Data_PreDeal(train_with_y,train2)
    #去除多余变量，train1只比train2多'y' 
    train1_ss_cut = train1_ss.drop('close',axis=1).copy()
    train2_ss_cut = train2_ss.drop('close',axis=1).copy()
    #提取主成分
    train1_ss_pca,train2_ss_pca = Get_Pca(train1_ss_cut,train2_ss_cut,-1)
    #变量选择
    trdata1 = train1_ss_pca.copy()
    trdata2 = train2_ss_pca.copy()

    var_use = Choose_Var(trdata1,max_var)
    train1_use = trdata1.loc[:,var_use].copy()
    train1_use = pd.concat([train1_use,train1_ss_pca.loc[:,'y']],axis=1)
    train2_use = trdata2.loc[:,var_use].copy()
    ####模型训练
    model = GDBT(train1_use)
    #####用模型进行预测，返回预测后的label
    pro,label = Deal_Proba(train2_use,model,threshold)
    train2_use_withy = Tag_Attach(train2_use,a,b)


    
######################################################debug

######################################################
import numpy as np
import pandas as pd
import os
import glob
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#取数据
path = u"F:\\工作\\立信\\cta\\data\\factor"
#path = "/Users/jsysley/Desktop/cta/data/factor"
#list_file = os.listdir(path)
list_file = List_File_Csv(path)
results = Read_All(path)
data_all = Deal_Factor(results,list_file)
data_all = data_all.loc['X2012.12.14_10.00.00':,:]
#分割数据集
train1 = data_all.iloc[:10000,:].copy()
train2 = data_all.iloc[10000:14000,:].copy()
test = data_all.iloc[14000:,:].copy()
#贴标签
train_with_y = Tag_Attach(train1,-0.005,0.006)
#返回标准化后的两个训练集
train1_ss,train2_ss = Data_PreDeal(train_with_y,train2)


    
    
    
    
    
