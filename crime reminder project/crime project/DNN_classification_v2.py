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
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler



train = pd.read_csv('/Users/calvin/python/crime project/test_Sample data (3d).csv')
#初始設定----------------------------------------------------------------------
train.dropna(inplace=True)
train_y=train['Event count']
#all_data = pd.concat([train, test], ignore_index = True)
train.drop(columns=["COPLNT_DAY"])
train_x=train
train_x.drop(columns=["Event count"])

train_x.dropna(axis=0, how='any')
train_x = pd.get_dummies(train_x)

#seperate the dataset----------------------------------------------------------
#case 1


train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.7)


#training set
train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
valid_x=valid_x.values
valid_y=valid_y.values
valid_y=valid_y.reshape(-1,1)


#標準化
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
valid_x = scaler.transform(valid_x)
#test = scaler.transform(test)   

#PCA---------------------------------------------------------------------------

pca_num=0
delta=0.1

pca=PCA(n_components = 0.999999)
train_x=pca.fit_transform(train_x)
valid_x=pca.transform(valid_x)
#test=pca.transform(test)
dimention=train_x.shape[1]

#Standarisation for whitening--------------------------------------------------

# to avoid roudoff error
# log-sum-exp trick----------------------------------------------------------
#case1
#getmin = np.min(train_y)
#getmax = np.max(train_y)
#train_yn = (train_y - getmin) / (getmax - getmin)
#valid_yn = (valid_y - getmin) / (getmax - getmin)

## case2
#train_y=np.log(train_y)
#valid_y=np.log(valid_y)

#DEEP LEARNING STRUCTURE-------------------------------------------------------

print("DNN start")
#learning_rate_setting=[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
learning_rate = 0.00001
training_epochs = 400
display_step = 20
batch_size=1024
layer=[300,250,200,180,150,150,100,50,10]
epsilon = 0.001
n_classes = 5
n_samples = train_x.shape[0]

def get_batch(data_x,data_y,batch_size):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y

def neural_net_model(X_data,input_dim):
    
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
        
# add Xavier initialization====================================================
# moving average for batch normalisaton
################        
    # layer input multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim,layer[0]])*np.sqrt(1/input_dim))
    b_1 = tf.Variable(tf.zeros([layer[0]]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
################

################
    # layer 1 multiplying and adding bias then activation function
    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([layer[0],layer[1]])*np.sqrt(1/layer[0]))
    b_2 = tf.Variable(tf.zeros([layer[1]]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    fc_mean, fc_var = tf.nn.moments(layer_2,axes=[0])
    scale_2 = tf.Variable(tf.ones([layer[1]]))
    shift_2 = tf.Variable(tf.zeros([layer[1]]))
    mean, var = mean_var_with_update()
    layer_2 = tf.nn.batch_normalization(layer_2, fc_mean, fc_var, shift_2, scale_2, epsilon)

    layer_2 = tf.nn.relu(layer_2)
################
    
################    
    # layer 2 multiplying and adding bias then activation function
    # layer 2 multiplying and adding bias then activation function
    W_3 = tf.Variable(tf.random_uniform([layer[1],layer[2]])*np.sqrt(1/layer[1]))
    b_3 = tf.Variable(tf.zeros([layer[2]]))
    layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
    fc_mean, fc_var = tf.nn.moments(layer_3,axes=[0])
    scale_3 = tf.Variable(tf.ones([layer[2]]))
    shift_3 = tf.Variable(tf.zeros([layer[2]]))
    mean, var = mean_var_with_update()
    layer_3 = tf.nn.batch_normalization(layer_3, fc_mean, fc_var, shift_3, scale_3, epsilon)
    layer_3 = tf.nn.relu(layer_3)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_4 = tf.Variable(tf.random_uniform([layer[2],layer[3]])*np.sqrt(1/layer[2]))
    b_4 = tf.Variable(tf.zeros([layer[3]]))
    layer_4 = tf.add(tf.matmul(layer_3,W_4), b_4)
    fc_mean, fc_var = tf.nn.moments(layer_4,axes=[0])
    scale_4 = tf.Variable(tf.ones([layer[3]]))
    shift_4 = tf.Variable(tf.zeros([layer[3]]))
    mean, var = mean_var_with_update()
    layer_4 = tf.nn.batch_normalization(layer_4, fc_mean, fc_var, shift_4, scale_4, epsilon)

    layer_4 = tf.nn.relu(layer_4)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_5 = tf.Variable(tf.random_uniform([layer[3],layer[4]])*np.sqrt(1/layer[3]))
    b_5 = tf.Variable(tf.zeros([layer[4]]))
    layer_5 = tf.add(tf.matmul(layer_4,W_5), b_5)
    fc_mean, fc_var = tf.nn.moments(layer_5,axes=[0])
    scale_5 = tf.Variable(tf.ones([layer[4]]))
    shift_5 = tf.Variable(tf.zeros([layer[4]]))
    mean, var = mean_var_with_update()
    layer_5 = tf.nn.batch_normalization(layer_5, fc_mean, fc_var, shift_5, scale_5, epsilon)

    layer_5 = tf.nn.relu(layer_5)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_6 = tf.Variable(tf.random_uniform([layer[4],layer[5]])*np.sqrt(1/layer[4]))
    b_6 = tf.Variable(tf.zeros([layer[5]]))
    layer_6 = tf.add(tf.matmul(layer_5,W_6), b_6)
    fc_mean, fc_var = tf.nn.moments(layer_6,axes=[0])
    scale_6 = tf.Variable(tf.ones([layer[5]]))
    shift_6 = tf.Variable(tf.zeros([layer[5]]))
    mean, var = mean_var_with_update()
    layer_6 = tf.nn.batch_normalization(layer_6, fc_mean, fc_var, shift_6, scale_6, epsilon)

    layer_6 = tf.nn.relu(layer_6)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_7 = tf.Variable(tf.random_uniform([layer[5],layer[6]])*np.sqrt(1/layer[5]))
    b_7 = tf.Variable(tf.zeros([layer[6]]))
    layer_7 = tf.add(tf.matmul(layer_6,W_7), b_7)
    fc_mean, fc_var = tf.nn.moments(layer_7,axes=[0])
    scale_7 = tf.Variable(tf.ones([layer[6]]))
    shift_7 = tf.Variable(tf.zeros([layer[6]]))
    mean, var = mean_var_with_update()
    layer_7 = tf.nn.batch_normalization(layer_7, fc_mean, fc_var, shift_7, scale_7, epsilon)

    layer_7 = tf.nn.relu(layer_7)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_8 = tf.Variable(tf.random_uniform([layer[6],layer[7]])*np.sqrt(1/layer[6]))
    b_8 = tf.Variable(tf.zeros([layer[7]]))
    layer_8 = tf.add(tf.matmul(layer_7,W_8), b_8)
    fc_mean, fc_var = tf.nn.moments(layer_8,axes=[0])
    scale_8 = tf.Variable(tf.ones([layer[7]]))
    shift_8 = tf.Variable(tf.zeros([layer[7]]))
    mean, var = mean_var_with_update()
    layer_8 = tf.nn.batch_normalization(layer_8, fc_mean, fc_var, shift_8, scale_8, epsilon)

    layer_8 = tf.nn.relu(layer_8)
################
    
################
    # layer 2 multiplying and adding bias then activation function
    W_9 = tf.Variable(tf.random_uniform([layer[7],n_classes])*np.sqrt(1/layer[7]))
    b_9 = tf.Variable(tf.zeros(n_classes))
    prediction = tf.add(tf.matmul(layer_8,W_9), b_9)
    fc_mean, fc_var = tf.nn.moments(prediction,axes=[0])
    scale_9 = tf.Variable(tf.ones([layer[8]]))
    shift_9 = tf.Variable(tf.zeros([layer[8]]))
    mean, var = mean_var_with_update()
    prediction = tf.nn.batch_normalization(prediction, fc_mean, fc_var, shift_9, scale_9, epsilon)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=Y))
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return cost , prediction
################

cost_history_train = np.empty(shape=[1],dtype=float)
cost_history_val = np.empty(shape=[1],dtype=float)
cost_history = np.empty(shape=[1],dtype=float)
cost_history_plot=[]


X = tf.placeholder("float32",[None, dimention],name="my_x")
Y = tf.placeholder("float32",name="my_y")

# our mean squared error cost function
# Gradinent Descent optimiztion just discussed above for updating weights and biases
cost,prediction = neural_net_model(X,dimention)
correct = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))  #get the max reaction and decide to the prediction class.
accuracy = tf.reduce_mean(tf.cast(correct,'float'))# cast 表示將原來的data轉換為其他type
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    # Run the initializer
#    sess = tf.InteractiveSession()    
    init = tf.global_variables_initializer()
    sess.run(init)


    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in get_batch(train_x,train_y,batch_size):
            sess.run(optimizer, feed_dict={X: x, Y: y})
#            cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:x,Y:y}))

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            pred_valid = sess.run(cost,feed_dict={X:valid_x,Y:valid_y})
            pred_train = sess.run(cost,feed_dict={X:train_x,Y:train_y})           
            print('Number: %d epoch' % (epoch+1),'\n','valid cost: ' , pred_valid)
            print('train cost: ' , pred_train)
            
            accuracy_valid = sess.run(accuracy,feed_dict={X:valid_x,Y:valid_y})
            accuracy_train = sess.run(accuracy,feed_dict={X:train_x,Y:train_y})           
            print('valid Acc: ' , accuracy_valid)
            print('Train Acc: ' , accuracy_train)


            cost_history_val = np.append(cost_history,sess.run(cost,feed_dict={X:valid_x,Y:valid_y}))
            cost_history_train = np.append(cost_history,sess.run(cost,feed_dict={X:train_x,Y:train_y}))

#            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            
#            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
#                 "W=", sess.run(W),"b=", sess.run(b))

    print("Optimization Finished!")
#    print("Training cost=", training_cost,'\n')
    
    #learning rate監控
    cost_history_plot=np.append(cost_history_plot,cost_history)
    cost_history_forLR=cost_history_plot
    
#   show final accuracy
    pred_valid = sess.run(cost,feed_dict={X:valid_x,Y:valid_y})
    pred_train = sess.run(cost,feed_dict={X:train_x,Y:train_y})
    print('final valid cost:' , pred_valid)
    print('final train cost:' , pred_train)

    accuracy_valid = sess.run(accuracy,feed_dict={X:valid_x,Y:valid_y})
    accuracy_train = sess.run(accuracy,feed_dict={X:train_x,Y:train_y})        
    print('valid Acc: ' , accuracy_valid)
    print('Train Acc: ' , accuracy_train)
    
#   save
    shadow_var0_name = ema.average_name(mean)
    shadow_var1_name = ema.average_name(var)
    saver = tf.train.Saver({shadow_var0_name: mean, shadow_var1_name: var})

    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./crime_model")



#plot different learning rates figure
#cost_history_plot=np.array(cost_history_plot)
#ja.Plot(np.arange(len(cost_history_plot),cost_history_plot)
    
    
#=============================================================================
# 畫圖驗證cost, epoch, optimal point============================================
#=============================================================================
    plt.figure(4)
    fig, ax = plt.subplots()
    ax.plot(cost_history,'r')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Cost')
    A=np.array(cost_history)
    best_epoch=np.argmin(A)
    print('best_cost:',min(cost_history),'achieved at epoch:',best_epoch)
    plt.show()   
    plt.pause(0.1)   
    
    
# =============================================================================
# start for testing set
# =============================================================================


 
#print("start to test the data")
#test=np.array(test, dtype=np.float64)   
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    saver.restore(sess, "./house_test_01_model")  
#    
#    test_y = sess.run(pred, feed_dict={X: test})
#    test_y=test_y.flatten()
#    Id=np.array(Id)
##    test_y=pd.DataFrame({"SalePrice":test_y})   
#    
##    #還原PCA
##    test_y=pca.inverse_transform(test_y)    
##    #還原標準化
##    test_y=scaler.inverse_transform(test_y)
#
#    submission = pd.DataFrame(data={"Id":Id,"SalePrice":test_y}, index=[np.arange(1459)])
#    submission.to_csv("submission_house_result.csv", index=False)
      
    