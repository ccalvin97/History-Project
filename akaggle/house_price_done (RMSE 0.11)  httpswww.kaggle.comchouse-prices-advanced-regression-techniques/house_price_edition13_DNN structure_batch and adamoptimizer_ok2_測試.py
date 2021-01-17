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
plt.pause(0.1)

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


#all_data=all_data.drop(columns=["GarageYrBlt","GarageYrBlt_log"])

all_data=all_data.drop(["GarageYrBlt","GarageYrBlt_log"], axis=1)


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
#train_y = scaler.transform(train_y) #<---

getmin = np.min(train_y)
getmax = np.max(train_y)
train_yn = (train_y - getmin) / (getmax - getmin)
valid_yn = (valid_y - getmin) / (getmax - getmin)

valid_x = scaler.transform(valid_x)
test = scaler.transform(test)   


qq
#PCA---------------------------------------------------------------------------

pca_num=0
delta=0.1

pca=PCA(n_components = 0.70)
train_x=pca.fit_transform(train_x)
valid_x=pca.transform(valid_x)
test=pca.transform(test)
dimention=train_x.shape[1]


#DEEP LEARNING STRUCTURE-------------------------------------------------------

""" 參考:
https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
"""

#learning_rate_setting=[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
learning_rate = 0.005
training_epochs = 1000
display_step = 10
batch_size=16 
layer_1_num=300
layer_2_num=50
layer_3_num=200
layer_4_num=40
layer_5_num=100
layer_6_num=50
layer_7_num=10

n_samples = train_x.shape[0]

def get_batch(data_x,data_y,batch_size):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y

def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,layer_1_num]))
    b_1 = tf.Variable(tf.zeros([layer_1_num]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([layer_1_num,layer_2_num]))
    b_2 = tf.Variable(tf.zeros([layer_2_num]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_3 = tf.Variable(tf.random_uniform([layer_2_num,layer_3_num]))
    b_3 = tf.Variable(tf.zeros([layer_3_num]))
    layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
    layer_3 = tf.nn.relu(layer_3)
    # layer 2 multiplying and adding bias then activation function
    W_4 = tf.Variable(tf.random_uniform([layer_3_num,layer_4_num]))
    b_4 = tf.Variable(tf.zeros([layer_4_num]))
    layer_4 = tf.add(tf.matmul(layer_3,W_4), b_4)
    layer_4 = tf.nn.relu(layer_4)
    # layer 2 multiplying and adding bias then activation function
    W_5 = tf.Variable(tf.random_uniform([layer_4_num,layer_5_num]))
    b_5 = tf.Variable(tf.zeros([layer_5_num]))
    layer_5 = tf.add(tf.matmul(layer_4,W_5), b_5)
    layer_5 = tf.nn.relu(layer_5)
    # layer 2 multiplying and adding bias then activation function
    W_6 = tf.Variable(tf.random_uniform([layer_5_num,layer_6_num]))
    b_6 = tf.Variable(tf.zeros([layer_6_num]))
    layer_6 = tf.add(tf.matmul(layer_5,W_6), b_6)
    layer_6 = tf.nn.relu(layer_6)
    # layer 2 multiplying and adding bias then activation function
    W_7 = tf.Variable(tf.random_uniform([layer_6_num,layer_7_num]))
    b_7 = tf.Variable(tf.zeros([layer_7_num]))
    layer_7 = tf.add(tf.matmul(layer_6,W_7), b_7)
    layer_7 = tf.nn.relu(layer_7)
    # output later multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([layer_7_num,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_7,W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output


X = tf.placeholder("float32",[None, dimention],name="my_x")
Y = tf.placeholder("float32",name="my_y")

pred = neural_net_model(X,dimention)
cost = tf.sqrt(tf.reduce_sum(tf.pow(pred-Y, 2))/n_samples)
# our mean squared error cost function
# Gradinent Descent optimiztion just discussed above for updating weights and biases

cost_history = np.empty(shape=[1],dtype=float)
cost_history_plot=[]



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    # Run the initializer
#    sess = tf.InteractiveSession()    
    init = tf.global_variables_initializer()
    sess.run(init)


    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in get_batch(train_x,train_y,batch_size):
#            x = x.reshape(x.shape[0],batch_size)
#            x=np.transpose(x)
#            y = y.reshape(y.shape[0],1)
            sess.run(optimizer, feed_dict={X: x, Y: y})
#            cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:x,Y:y}))

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            cost_valid = sess.run(cost,feed_dict={X:valid_x,Y:valid_y})
            cost_train = sess.run(cost,feed_dict={X:train_x,Y:train_y})           
            print('Number: %d epoch' % (epoch+1),'\n','valid cost: ' , cost_valid)
            print('Train cost: ' , cost_train)
            cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:train_x,Y:train_y}))
#            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            
#            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
#                 "W=", sess.run(W),"b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training cost=", training_cost,'\n')
    
    #learning rate監控
    cost_history_plot=np.append(cost_history_plot,training_cost)
    
# cost history tracking
    plt.figure(2)    
    xx=np.arange(len(cost_history))*display_step
    plt.plot(xx,cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()
    plt.pause(0.1)
    
    pred_y = sess.run(pred, feed_dict={X: valid_x})
    rmse=tf.sqrt(tf.reduce_sum(tf.pow(pred_y-valid_y, 2))/n_samples)
    print("RMSE: %.4f" % sess.run(rmse)) 
    
    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./house_test_01_model")


    #視覺化 fit plot
    print("finish, plot the scatter figure")
    plt.figure(3)
    fig, ax = plt.subplots()
    ax.scatter(valid_y, pred_y)
    ax.plot([valid_y.min(), valid_y.max()], [valid_y.min(), valid_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()   
    plt.pause(0.1)

#plot different learning rates figure
#cost_history_plot=np.array(cost_history_plot)
#ja.Plot(np.arange(len(cost_history_plot)),cost_history_plot)
#

    
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
    
#    #還原PCA
#    test_y=pca.inverse_transform(test_y)    
#    #還原標準化
#    test_y=scaler.inverse_transform(test_y)

    submission = pd.DataFrame(data={"Id":Id,"SalePrice":test_y}, index=[np.arange(1459)])
    submission.to_csv("submission_house_result.csv", index=False)
    
    