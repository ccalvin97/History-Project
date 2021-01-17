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



# import dataset
train = pd.read_csv('/Users/calvin/python/car project/porsche_data.csv')
train=train.drop(["Vin"],axis=1)

#train=train[:3600]
#test=test.drop(test[:800].index,axis=0)

#def function(data, test):
# adjust data -> pull out label
train_y= train["Price"]
train_x= train.drop(["Price"],axis=1)
#test_y= test["Price"]
#test_x= test.drop(["Price"],axis=1)

# one hot encoding 
train_x=pd.get_dummies(train_x)
#test_x=pd.get_dummies(test_x)

# seperate the tesintg set from the big dataset
test_x=train_x.sample(n=500,random_state=0)
train=train_x.drop(train_x.sample(n=500,random_state=0).index,axis=0)

test_y=train_y.sample(n=500,random_state=0)
train=train_y.drop(train_y.sample(n=500,random_state=0).index,axis=0)

#seperate dataset to training set and validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3)

# fine tune the data type
train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
val_x=val_x.values
val_y=val_y.values
val_y=val_y.reshape(-1,1)
test_x=test_x.values
test_y=test_y.values
test_y=test_y.reshape(-1,1)

#Standarisation for whitening--------------------------------------------------
#standarisation
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

#train_y = scaler.transform(train_y) #<---  


#PCA---------------------------------------------------------------------------

pca_num=0
delta=0.1

pca=PCA(n_components = 0.999)
train_x=pca.fit_transform(train_x)
val_x=pca.transform(val_x)
test_x=pca.transform(test_x)
dimention=train_x.shape[1]


#DEEP LEARNING STRUCTURE-------------------------------------------------------

#calculate the processing time
import datetime
starttime = datetime.datetime.now()

#parameter 
print("DNN start")
learning_rate = 0.1
training_epochs = 200
display_step = 10
batch_size=256
layer=[600,512,450,400,350,300,256,200,128,100,64,32,8]
n_samples = train_x.shape[0]
dropout_rate=0.15

# set mini batch
def get_batch(data_x,data_y,batch_size):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y

# DNN start to run
def neural_net_model(X_data,input_dim):
    
    # setting moving average in order to accelerate the precessing in the testing set
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
    W_1 = tf.Variable(tf.random_normal([input_dim,layer[0]],mean=0.0,stddev=0.01)*np.sqrt(1/input_dim))
    W_1 = tf.layers.dropout(W_1, rate=dropout_rate, training=tf_is_training)  
    b_1 = tf.Variable(tf.zeros([layer[0]]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
################

################
    # layer 2 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_normal([layer[0],layer[1]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[0]))
    W_2 = tf.layers.dropout(W_2, rate=dropout_rate, training=tf_is_training)  
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
    # layer 3 multiplying and adding bias then activation function
    W_3 = tf.Variable(tf.random_normal([layer[1],layer[2]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[1]))
    W_3 = tf.layers.dropout(W_3, rate=dropout_rate, training=tf_is_training)  

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
    # layer 4 multiplying and adding bias then activation function
    W_4 = tf.Variable(tf.random_normal([layer[2],layer[3]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[2]))
    W_4 = tf.layers.dropout(W_4, rate=dropout_rate, training=tf_is_training)  
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
    # layer 5 multiplying and adding bias then activation function
    W_5 = tf.Variable(tf.random_normal([layer[3],layer[4]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[3]))
    W_5 = tf.layers.dropout(W_5, rate=dropout_rate, training=tf_is_training)  
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
    # layer 6 multiplying and adding bias then activation function
    W_6 = tf.Variable(tf.random_normal([layer[4],layer[5]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[4]))
    W_6 = tf.layers.dropout(W_6, rate=dropout_rate, training=tf_is_training)  
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
    # layer 7 multiplying and adding bias then activation function
    W_7 = tf.Variable(tf.random_normal([layer[5],layer[6]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[5]))
    W_7 = tf.layers.dropout(W_7, rate=dropout_rate, training=tf_is_training)  
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
    # layer 8 multiplying and adding bias then activation function
    W_8 = tf.Variable(tf.random_normal([layer[6],layer[7]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[6]))
    W_8 = tf.layers.dropout(W_8, rate=dropout_rate, training=tf_is_training)  
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
    # layer 9 multiplying and adding bias then activation function
    W_9 = tf.Variable(tf.random_normal([layer[7],layer[8]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[7]))
    W_9 = tf.layers.dropout(W_9, rate=dropout_rate, training=tf_is_training)  
    b_9 = tf.Variable(tf.zeros([layer[8]]))
    layer_9 = tf.add(tf.matmul(layer_8,W_9), b_9)
    fc_mean, fc_var = tf.nn.moments(layer_9,axes=[0])
    scale_9 = tf.Variable(tf.ones([layer[8]]))
    shift_9 = tf.Variable(tf.zeros([layer[8]]))
    mean, var = mean_var_with_update()
    layer_9 = tf.nn.batch_normalization(layer_9, fc_mean, fc_var, shift_9, scale_9, epsilon)

    layer_9= tf.nn.relu(layer_9)
################
################
    # layer 10 multiplying and adding bias then activation function
    W_10 = tf.Variable(tf.random_normal([layer[8],layer[9]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[8]))
    W_10 = tf.layers.dropout(W_10, rate=dropout_rate, training=tf_is_training)  
    b_10 = tf.Variable(tf.zeros([layer[9]]))
    layer_10 = tf.add(tf.matmul(layer_9,W_10), b_10)
    fc_mean, fc_var = tf.nn.moments(layer_10,axes=[0])
    scale_10 = tf.Variable(tf.ones([layer[9]]))
    shift_10 = tf.Variable(tf.zeros([layer[9]]))
    mean, var = mean_var_with_update()
    layer_10 = tf.nn.batch_normalization(layer_10, fc_mean, fc_var, shift_10, scale_10, epsilon)

    layer_10= tf.nn.relu(layer_10)
################
################
    # layer 11 multiplying and adding bias then activation function
    W_11 = tf.Variable(tf.random_normal([layer[9],layer[10]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[9]))
    W_11 = tf.layers.dropout(W_11, rate=dropout_rate, training=tf_is_training)  
    b_11 = tf.Variable(tf.zeros([layer[10]]))
    layer_11 = tf.add(tf.matmul(layer_10,W_11), b_11)
    fc_mean, fc_var = tf.nn.moments(layer_11,axes=[0])
    scale_11 = tf.Variable(tf.ones([layer[10]]))
    shift_11 = tf.Variable(tf.zeros([layer[10]]))
    mean, var = mean_var_with_update()
    layer_11 = tf.nn.batch_normalization(layer_11, fc_mean, fc_var, shift_11, scale_11, epsilon)

    layer_11= tf.nn.relu(layer_11)
################
    ################
    # layer 12 multiplying and adding bias then activation function
    W_12 = tf.Variable(tf.random_normal([layer[10],layer[11]],mean=0.0,stddev=0.01)*np.sqrt(1/layer[10]))
    W_12 = tf.layers.dropout(W_12, rate=dropout_rate, training=tf_is_training)  
    b_12 = tf.Variable(tf.zeros([layer[11]]))
    layer_12 = tf.add(tf.matmul(layer_11,W_12), b_12)
    fc_mean, fc_var = tf.nn.moments(layer_12,axes=[0])
    scale_12 = tf.Variable(tf.ones([layer[11]]))
    shift_12 = tf.Variable(tf.zeros([layer[11]]))
    mean, var = mean_var_with_update()
    layer_12 = tf.nn.batch_normalization(layer_12, fc_mean, fc_var, shift_12, scale_12, epsilon)

    layer_12= tf.nn.relu(layer_12)
################
################
    # layer 13 multiplying and adding bias then activation function
    W_13 = tf.Variable(tf.random_normal([layer[11],1],mean=0.0,stddev=0.01)*np.sqrt(1/layer[11]))
    W_13 = tf.layers.dropout(W_13, rate=dropout_rate, training=tf_is_training)  
    b_13 = tf.Variable(tf.zeros(1))
    output = tf.add(tf.matmul(layer_12,W_13), b_13)


    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    return output





#set the tensorflow variable first
X = tf.placeholder("float32",[None, dimention],name="my_x")
Y = tf.placeholder("float32",name="my_y")
tf_is_training = tf.placeholder(tf.bool, None) 

#cost function
pred = neural_net_model(X,dimention)
cost = tf.sqrt(tf.reduce_sum(tf.pow(pred-Y, 2))/n_samples)
# our mean squared error cost function
# Gradinent Descent optimiztion just discussed above for updating weights and biases

cost_history = np.empty(shape=[1],dtype=float)
cost_history_plot=[]


#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

cost_history_train=[]
cost_history_valid=[]
with tf.Session() as sess:
    # Run the initializerPlaceholder_2
#    sess = tf.InteractiveSession()    
    init = tf.global_variables_initializer()
    sess.run(init)


    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in get_batch(train_x,train_y,batch_size):
            sess.run(optimizer, feed_dict={X: x, Y: y,tf_is_training: True})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y,tf_is_training: False})
            cost_valid = sess.run(cost,feed_dict={X:val_x,Y:val_y,tf_is_training: False})
            cost_train = sess.run(cost,feed_dict={X:train_x,Y:train_y,tf_is_training: False})           
            print('Number: %d epoch' % (epoch+1),'\n','valid cost: ' , cost_valid)
            print('Train cost: ' , cost_train)
            cost_history_train = np.append(cost_history_train,sess.run(cost,feed_dict={X:train_x,Y:train_y,tf_is_training: False}))
            cost_history_valid = np.append(cost_history_valid,sess.run(cost,feed_dict={X:val_x,Y:val_y,tf_is_training: False}))

    print("Optimization Finished!  Start for the validation")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y,tf_is_training: False})
    cost_valid = sess.run(cost,feed_dict={X:val_x,Y:val_y,tf_is_training: False})
    print("Training cost=", training_cost,'\n')
    print("Validation cost=", cost_valid,'\n')
    #learning rate監控
    cost_history_plot=np.append(cost_history_plot,training_cost)
    

    pred_y = sess.run(pred, feed_dict={X: val_x,tf_is_training: False})
    rmse=tf.sqrt(tf.reduce_sum(tf.pow(pred_y-val_y, 2))/n_samples)
    print("RMSE: %.4f" % sess.run(rmse)) 
    
    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./car_project_model")

#================================plot==========================================
# cost visualisation for training set and validation set
plt.figure(2)
xx=np.arange(len(cost_history_train))*display_step
plt.plot(xx,cost_history_train,label='train line')
valid_plot=np.arange(len(cost_history_valid))*display_step
plt.plot(valid_plot,cost_history_valid,label='val line')
plt.legend(loc='upper right')
plt.show()
plt.pause(0.1)
    
#visualisation fit plot
print("finish, plot the scatter figure")
plt.figure(3)
fig, ax = plt.subplots()
ax.scatter(val_y, pred_y)
ax.plot([val_y.min(), val_y.max()], [val_y.min(), val_y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.ylim((0,250000))
plt.xlim((0,250000))
plt.show()   
plt.pause(0.1)
    
# calculate the processing time
endtime = datetime.datetime.now()
print ("the processing time is :",(endtime - starttime).seconds,"seconds") 

# =============================================================================
# start for testing set
# =============================================================================
import datetime
starttime = datetime.datetime.now()

n_samples_test=test_x.shape[0]
 
print("start to test the data")
#test=np.array(test, dtype=np.float64)   
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./car_project_model")  
    pred_y = sess.run(pred, feed_dict={X: test_x,tf_is_training: False})

    cost_test = sess.run(cost,feed_dict={X:test_x,Y:test_y,tf_is_training: False})  
    
    rmse=tf.sqrt(tf.reduce_sum(tf.pow(pred_y-test_y, 2))/n_samples_test)
    print("RMSE: %.4f" % sess.run(rmse))

    #visualisation fit plot
    print("finish, plot the scatter figure")
    plt.figure(4)
    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    plt.ylim((0,250000))
    plt.xlim((0,250000))
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()   
    plt.pause(0.1)
    
endtime = datetime.datetime.now()
print ("the processing time is :",(endtime - starttime).seconds,"seconds")  
    
    
