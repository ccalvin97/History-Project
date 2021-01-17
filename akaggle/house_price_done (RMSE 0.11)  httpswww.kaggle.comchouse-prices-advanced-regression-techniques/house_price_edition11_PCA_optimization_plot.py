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

from scipy.stats import norm, skew #for some statistics
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
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.3)

#training set
train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
valid_x=valid_x.values
valid_y=valid_y.values
valid_y=valid_y.reshape(-1,1)





#plt.figure(1)
#plt.scatter(x="GarageYrBlt", y="SalePrice",data=all_data)
#plt.show()
#plt.pause(0.1)

#標準化
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
valid_x = scaler.transform(valid_x)
test = scaler.transform(test)   

#PCA---------------------------------------------------------------------------
pca_ratio=0.99999
pca_num=0
delta=0.005
rmse_history=[]
range_num=4
pca_ratio_history=[]

for pca_num in np.arange(range_num):
    print("第%d圈,pca調整至 %f" % (pca_num+1, pca_ratio))
    pca=PCA(n_components=pca_ratio)
    train_x=pca.fit_transform(train_x)
    valid_x=pca.transform(valid_x)
    test=pca.transform(test) 
    dimention=train_x.shape[1]
    pca_ratio_history = np.append(pca_ratio_history,pca_ratio)
    pca_ratio=pca_ratio-delta
    #testing set
    
    
    #建模code
    rng = np.random
    
    # Parameters
    learning_rate = 100
    training_epochs = 200
    display_step = 10
    
    n_samples = train_x.shape[0]
    
    # tf Graph Input
    X = tf.placeholder("float64",[None, dimention],name="my_x")
    Y = tf.placeholder("float64",name="my_y")
    
    # Set model weights
    W=tf.Variable(tf.truncated_normal([dimention, 1], stddev=0.1),name="weight" )
    W=tf.cast(W, tf.float64)
    b = tf.Variable(rng.rand(), name="bias")
    b=tf.cast(b, tf.float64)
    
    
    # Construct a linear model
    a=tf.matmul( X,W )
    a=tf.cast(a, tf.float64)
    pred=tf.add(a, b)
    pred=tf.cast(pred, tf.float64)
    
    cost = tf.sqrt(tf.reduce_sum(tf.pow(pred-Y, 2))/n_samples)
    #cost= tf.sqrt(cost)
    #hypothesis = tf.sigmoid(tf.multiply(X, W) + b)
    #cost=tf.reduce_mean(Y *( tf.log(cost + 1e-4)) + (1 - Y) * (tf.log(1 - cost+1e-4))) 
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    cost_history = np.empty(shape=[1],dtype=float)
    
    # Start training, reset all variable in seesion, including bias and weight
    sess = tf.InteractiveSession()

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            x = x.reshape(x.shape[0],1)
            x=np.transpose(x)
            y = y.reshape(y.shape[0],1)
            sess.run(optimizer, feed_dict={X: x, Y: y})
#            cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:x,Y:y}))

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:train_x,Y:train_y}))
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            
            
#            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
#                 "W=", sess.run(W),"b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training cost=", training_cost, '\n')

# cost history tracking
    xx=np.arange(len(cost_history))*display_step
    plt.plot(xx,cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.pause(0.1)
    plt.show()

    pred_y = sess.run(pred, feed_dict={X: valid_x})
    rmse=tf.sqrt(tf.reduce_sum(tf.pow(pred_y-valid_y, 2))/n_samples)
    rmse_history=np.append(rmse_history, sess.run(rmse))
    print("RMSE: %.4f" % sess.run(rmse)) 
    
    
#    plt.plot(valid_x, valid_y, 'bo', label='Testing data')
#    plt.plot(train_x, train_x * sess.run(W) + sess.run(b), label='Fitted line')
#    plt.legend()
#    plt.show()

  
    #畫圖-pca參數 
    if pca_num+1 == range_num :     
        print("PCA variable plot")
        fig, ax = plt.subplots()
        ax.scatter(pca_ratio_history,rmse_history)
        ax.set_xlabel('pca_ratio_history')
        ax.set_ylabel('RMSE')
        plt.xlim((pca_ratio_history.min(), pca_ratio_history.max()))
        plt.ylim((rmse_history.min(), rmse_history.max()))
        plt.show()
        
        
        
    #plot data散布圖和最佳曲線    
    if pca_num+1 == range_num :        
        print("finish, plot the scatter figure")
        fig, ax = plt.subplots()
        ax.scatter(valid_y, pred_y)
        ax.plot([valid_y.min(), valid_y.max()], [valid_y.min(), valid_y.max()], 'k--', lw=3)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()   
        
        print(rmse_history)

#store module------------------------------------------------------------------
    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./house_test_01_model")

# start for testing set--------------------------------------------------------
   
print("start to test the data")
test=np.array(test, dtype=np.float64)   
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./house_test_01_model")  
    
    test_y = sess.run(pred, feed_dict={X: test})
    test_y=test_y.flatten()
    Id=np.array(Id)
#    test_y=pd.DataFrame({"SalePrice":test_y})   
    
    submission = pd.DataFrame(data={"Id":Id,"SalePrice":test_y}, index=[np.arange(1459)])
    submission.to_csv("submission_house_result.csv", index=False)
      
    
        # Graphic display
#    plt.plot(train_x, train_y, 'ro', label='Original data')
#    cost_plot=np.dot( train_x,sess.run(W) )
#    plt.plot(train_x, cost_plot, label='Fitted line')
#    plt.legend()
#    plt.pause(0.1)
#    plt.show()
        
#    mean=np.mean(train_x)
#    std=np.std(train_x)
#    print(mean)
#    print(std)