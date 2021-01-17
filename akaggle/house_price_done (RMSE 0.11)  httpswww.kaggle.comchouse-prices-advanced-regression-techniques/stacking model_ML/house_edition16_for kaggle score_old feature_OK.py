# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
import  tensorflow as tf
from pandas.core.frame import DataFrame
import JA_house as ja
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')




train = pd.read_csv('house_price_train.csv')
test = pd.read_csv('house_price_test.csv')

#合併前處理---------------------------------------------------------------------
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index,inplace=True)
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,inplace=True)
train.drop(train[(train['YearBuilt']<1900) & (train['SalePrice']>400000)].index,inplace=True)
train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index,inplace=True)
train.reset_index(drop=True, inplace=True)


#初始設定----------------------------------------------------------------------
Id=test['Id']
all_data = pd.concat([train, test], ignore_index = True)

feature=list(train.columns)
label="SalePrice"
feature.remove("SalePrice")

#將label 換到第一行
SalePrice=all_data.pop("SalePrice")
all_data.insert(0,"SalePrice",SalePrice)


#data cleaning ----------------------------------------------------------------

 
#填充缺失值
def fill_missings(res):

    res['Alley'] = res['Alley'].fillna('missing')
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])
    res['MasVnrType'] = res['MasVnrType'].fillna('None')
    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])
    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])
    res['GarageType'] = res['GarageType'].fillna('missing')
    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])
    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])
    res['GarageCond'] = res['GarageCond'].fillna('missing')
    res['Fence'] = res['Fence'].fillna('missing')
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])    
    res['Utilities'] = res['Utilities'].fillna('missing')
    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])    
    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])
    res["Functional"] = res["Functional"].fillna("Typ")
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
 #   res['SaleCondition'] = res['SaleCondition'].fillna('missing')
    #数值型变量的空值先用0值替换
    flist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                     'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                     'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                     'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
    #0值替换   
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)    
    return res
all_data=fill_missings(all_data)

##數值特徵無意義，轉成str
#all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#all_data['YrSold'] = all_data['YrSold'].astype(str)
#all_data['MoSold'] = all_data['MoSold'].astype(str)
#all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#順序特徵編碼
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

def QualToInt(x):
    if(x=='Ex'):
        r = 0
    elif(x=='Gd'):
        r = 1
    elif(x=='TA'):
        r = 2
    elif(x=='Fa'):
        r = 3
    elif(x=='missing'):
        r = 4
    else:
        r = 5
    return r

all_data['ExterQual'] = all_data['ExterQual'].apply(QualToInt)
all_data['ExterCond'] = all_data['ExterCond'].apply(QualToInt)
all_data['KitchenQual'] = all_data['KitchenQual'].apply(QualToInt)
all_data['HeatingQC'] = all_data['HeatingQC'].apply(QualToInt)
all_data['BsmtQual'] = all_data['BsmtQual'].apply(QualToInt)
all_data['BsmtCond'] = all_data['BsmtCond'].apply(QualToInt)
all_data['FireplaceQu'] = all_data['FireplaceQu'].apply(QualToInt)
all_data['GarageQual'] = all_data['GarageQual'].apply(QualToInt)
all_data['PoolQC'] = all_data['PoolQC'].apply(QualToInt)
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].apply(QualToInt)
all_data['MasVnrType'] = all_data['MasVnrType'].apply(QualToInt)
all_data['Foundation'] = all_data['Foundation'].apply(QualToInt)
all_data['Functional'] = all_data['Functional'].apply(QualToInt)
all_data['HouseStyle'] = all_data['HouseStyle'].apply(QualToInt)
all_data['BsmtExposure'] = all_data['BsmtExposure'].apply(QualToInt)
all_data['PavedDrive'] = all_data['PavedDrive'].apply(QualToInt)
all_data['Street'] = all_data['Street'].apply(QualToInt)


#新增特徵
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['HasWoodDeck'] = (all_data['WoodDeckSF'] == 0) * 1
all_data['HasOpenPorch'] = (all_data['OpenPorchSF'] == 0) * 1
all_data['HasEnclosedPorch'] = (all_data['EnclosedPorch'] == 0) * 1
all_data['Has3SsnPorch'] = (all_data['3SsnPorch'] == 0) * 1
all_data['HasScreenPorch'] = (all_data['ScreenPorch'] == 0) * 1

all_data['YearsSinceRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)

all_data['Total_Home_Quality'] = all_data['OverallQual'].astype(int) + all_data['OverallCond'].astype(int)

#數據轉換
quantitative = [f for f in train.columns if train.dtypes[f] != 'object'and  train.dtypes[f] != 'str']
quantitative.remove('SalePrice')
f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
plt.show()
plt.pause(0.1)

skewed_feats = all_data[quantitative].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness .head(20)

def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res
loglist=skewness[abs(skewness)>0.15].index.tolist()
all_data = addlogs(all_data, loglist)

## print出label對於其他所有變數的相關係數
#ja.Plot_Corr_Label(label, all_data)
#
##檢查缺值以及缺值的比例並刪除某些變數很少的丟失值
#all_data=ja.Checkdelete_Na(feature,label, all_data)   
#
#cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
#for col in cols1:
#    all_data[col].fillna("None", inplace=True)
#    
#cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
#for col in cols:
#    all_data[col].fillna(0, inplace=True)
#    
#all_data['LotFrontage']=all_data.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))




#去除id欄位
#all_data=all_data.drop(["Id"], axis=1)

# feature engineering----------------------------------------------------------

all_data=pd.get_dummies(all_data)
#ja.Plot_Corr_01(all_data)
#ja.Plot_Corr_Matrix(all_data)
#feature2=list(all_data.columns)
#for i in feature2:
#    ja.outlier(i,label,all_data)



# 建模start
#------------------------------------------------------------------------------
#A linear regression learning algorithm example using TensorFlow library.
#from __future__ import print_function


all_data=all_data.drop(columns=["GarageYrBlt","GarageYrBlt_log"])

ja.Checkde_Na(list(all_data.columns), all_data)

# 調整data，準備開始建模
train=all_data[all_data[label].notnull()]
test=all_data[all_data[label].isnull()].drop(label,axis=1)
train_y=train[label]
train_x=train.drop([label],axis=1)


#------------------------------------------------------------------------------

#training set
train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)





#plt.figure(1)
#plt.scatter(x="GarageYrBlt", y="SalePrice",data=all_data)
#plt.show()
#plt.pause(0.1)

#標準化------------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test = scaler.transform(test)   


#PCA---------------------------------------------------------------------------

pca_num=0
delta=0.1

pca=PCA(n_components = 0.999999)
train_x=pca.fit_transform(train_x)
test=pca.transform(test)
dimention=train_x.shape[1]

#轉換變數名稱
test_X_scaled=test
X_scaled=train_x
y= train.SalePrice

#pipe  ==  Pipeline([
#    ('labenc', labelenc()),
#    ('add_feature', add_feature(additional=2)),
#    ('skew_dummies', skew_dummies(skew=1)),
#    ])
#    
#full_pipe = pipe.fit_transform(full)
#n_train=train.shape[0]
#X = full_pipe[:n_train]
#test_X = full_pipe[n_train:]
#y= train.SalePrice
#
#X_scaled = scaler.fit(X).transform(X)
y_log = np.log(y)
#test_X_scaled = scaler.transform(test_X)  
#  
#pca = PCA(n_components=350)
#X_scaled=pca.fit_transform(X_scaled)
#test_X_scaled = pca.transform(test_X_scaled)



#Modeling & Evaluation---------------------------------------------------------
# machine learning scihit tutorial->  http://scikit-learn.org/stable/tutorial/basic/tutorial.html
""" 1. 定義cost function
"""

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
#cross_val_score->交叉驗證功能, cv=5表示dataset分五份交換做測試， 負號因為scoring 本身帶負號目標要取正值
# https://blog.csdn.net/xiaodongxiexie/article/details/71915259

""" 2. 測試演算法
"""
#models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
#          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
#          ExtraTreesRegressor(),XGBRegressor()]

#names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]

#for name, model in zip(names, models):
#    score = rmse_cv(model, X_scaled, y_log)
#    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
#參考:http://kuanghy.github.io/2016/11/25/python-str-format

#超參數最佳化-------------------------------------------------------------------
""" 3. 最佳化超參數
"""
#class grid():
#    def __init__(self,model):
#        self.model = model
#    
#    def grid_get(self,X,y,param_grid):
#        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
#        grid_search.fit(X,y)
#        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
#        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
#        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#        
#
#grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})
#grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
#grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
#param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}
#grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
#grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],'max_iter':[10000]})


#Ensemble Methods--------------------------------------------------------------
#class AverageWeight(BaseEstimator, RegressorMixin):
#    def __init__(self,mod,weight):
#        self.mod = mod
#        self.weight = weight
#        
#    def fit(self,X,y):
#        self.models_11 = [clone(x) for x in self.mod]
#        for model in self.models_11:
#            model.fit(X,y)
#        return self
#    
#    def predict(self,X):
#        w = list()
#        pred = np.array([model.predict(X) for model in self.models_11])
#        # for every data point, single model prediction times weight, then add them together
#        for data in range(pred.shape[1]):
#            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
#            w.append(np.sum(single))
#        return w
#clone-> It yields a new estimator with the same parameters that has not been fit on any data.
        
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()

# assign weights based on their gridsearch score
#w1 = 0.02
#w2 = 0.2
#w3 = 0.25
#w4 = 0.3
#w5 = 0.03
#w6 = 0.2
#
#weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
#rmse_cv(weight_avg,X_scaled,y_log),  rmse_cv(weight_avg,X_scaled,y_log).mean()
#weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
#rmse_cv(weight_avg,X_scaled,y_log),  rmse_cv(weight_avg,X_scaled,y_log).mean()



#Stacking----------------------------------------------------------------------
#參考: https://medium.com/@morris_tai/tatanic%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%A8%A1%E5%9E%8Bstacking%E6%95%B4%E7%90%86-523884f3bb98

class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
    
# split(x,y) -> Generate indices to split data into training and test set.  
#KFold: training set 切成若干分並交換跑training
#n_splits: training set切成幾份， random_state:種子, shuffle: 每次取是否洗牌?            
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
#clone(x)-> It yields a new estimator with the same parameters that has not been fit on any data.
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标    

a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
#ravel() 功能等同 numpy.flatten()


""" get_oof 方法   PS.方法1 和 方法2 結果相同
"""
#print("submission start for method 1")
#stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
#X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
#ker.fit(X_train_stack, y_log)
#pred = ker.predict(X_test_stack)
#pred2=np.exp(pred)
#print(rmse_cv(stack_model,X_train_stack,b))
#
#result=pd.DataFrame({'Id':Id, 'SalePrice':pred2})
#result.to_csv("submission0830_2.csv",index=False)


""" fit, predict 方法
"""

print("submission start for method 2")
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
stack_model.fit(a,b)
pred2 = np.exp(stack_model.predict(test_X_scaled))

result=pd.DataFrame({'Id':Id, 'SalePrice':pred2})
result.to_csv("submission0830.csv",index=False)
