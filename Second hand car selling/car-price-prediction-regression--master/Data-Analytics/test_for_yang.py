#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
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
#from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split



train = pd.read_csv('/Users/calvin/python/car project/porsche_data.csv')

test = pd.read_csv('/Users/calvin/python/car project/porsche_testing.csv')




#-----function start------------------------------------
def function(train,test):
    train=train.drop(["Vin"],axis=1)
    test=test.drop(["Vin"],axis=1)
    train=pd.concat([train, test],ignore_index=True)
    
    
    #def function(data, test):
        
    train_y= train["Price"]
    train_x= train.drop(["Price"],axis=1)
    #test_y= test["Price"]
    #test_x= test.drop(["Price"],axis=1)
    
    train_x=pd.get_dummies(train_x)
    #test_x=pd.get_dummies(test_x)
    
    #train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)
    
    #train_x=train_x.values
    train_y=train_y.values
    train_y=train_y.reshape(-1,1)
    #test_x=test_x.values
    #test_y=test_y.values
    #test_y=test_y.reshape(-1,1)
    
    
    #test_y=test_y.astype(np.int)
    train_y=train_y.astype(np.int)
    
    
    #test_x=np.array(test_x)
    #test_x=test_x.astype(np.float)
    # 標準化
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    #test_x=scaler.transform(test_x)
    
    
    #PCA---------------------------------------------------------------------------
    pca_num=0
    delta=0.1
    pca=PCA(n_components = 0.999999)
    train_x=pca.fit_transform(train_x)
    #test_x=pca.transform(test_x)
    
    dimension=train_x.shape[1]
    
    
    
    degree=6
    i=0
    train_x_copy=train_x
    for power in range(2,degree+1):
        i=i+1
        train_x=np.concatenate((train_x, np.power(train_x_copy,power)),axis=1) 
    # divide data in to training and testing set
    test_x=train_x[-1,:]
    test_y=train_y[-1,:]
    train_x=train_x[:-1]
    train_y=train_y[:-1]
    
    test_y=test_y.reshape(-1,1)
    test_x=test_x.reshape(1,-1)
    
    
    #train_x_copy=train_x
    #test_x_copy=test_x
    #for power in range(2,degree+1):
    #    i=i+1
    #    train_x=np.concatenate((train_x, np.power(train_x_copy,power)),axis=1) 
    #    test_x=np.concatenate((test_x, np.power(test_x_copy,power)),axis=1)
    #kernelridge
    from sklearn.linear_model import RidgeCV
    K=5
    kf = KFold(n_splits=K)
    RR = RidgeCV(alphas=np.logspace(-3, 3, 100))
    i=0
    rmse=0
    #for train_index, test_index in kf.split(train_x):
    #    RR.fit(train_x[train_index],train_y[train_index])
    #    y_RR = RR.predict(train_x[test_index])
    #
    #    plt.scatter(np.ones(y_RR.shape)*i,y_RR)
    #    i=i+1
    #    mse=np.mean((np.round(y_RR) != train_y[test_index])**2)
    #    rmse+=np.sqrt(mse)/K
    #    y_RR2 = RR.predict (test_x)
    
    
    RR.fit(train_x,train_y)
    y_RR3 = RR.predict (test_x)
    return y_RR3

y_RR45=function(train,test)