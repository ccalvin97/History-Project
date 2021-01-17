import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Input data

train_data = pd.read_csv(r"Data/train.csv")
test_data = pd.read_csv(r"Data/test.csv")


#Feature Engineering
from sklearn.preprocessing import Imputer

def nan_padding(data, columns):
    for column in columns:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
# reshape(-1,1) 表示列數max行數為1
#       1: imputer is helping to fill some nan value in certain column you select
#       2: In order to get the process, you need to reshape your column to (-1,1), which means the (raw, col) , raw is
#          raw is uncertain, but we make the col to 1 
#       3: about the fit_transform:
#          When you fill the data with cetain mean value, we apply certain formula with certain parameters
#          To using the same parameter when filling the nana value in  TEST_DATA_SET and TRAIN_DATA_SET, 
#          The fit_transform , simply combine Fit() and Transform() together.
#           At first, we launch fit() once and save the parameter internally, some when you applying the other dataset
#           The program will automatically read the parameter from previous Fit() then transform() other DataSet
#           Johnny, Hung, J&A
    return data



nan_columns = ["Age", "SibSp", "Parch"]
train_data = nan_padding(train_data,nan_columns)
test_data = nan_padding(test_data,nan_columns)


#save passengerid for evalulation
test_passenger_id=test_data["PassengerId"]

def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId","Name","Ticket","Fare","Cabin","Embarked"]
train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

#You can TEST with train.data.head()


#dummy Pclass to Pclcass_1, Pclass_2, Pclass_3
#可以新增多行，每行都代表一種feature的結果
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1) #Create Pclass_1, pclass_2 etc..
        data = data.drop(column, axis=1) #Drop original Pclass
    return data


dummy_columns = ["Pclass"]
train_data = dummy_data(train_data, dummy_columns)
test_data = dummy_data(test_data, dummy_columns)


#SEX to Int     'male' -> 1, 'female' -> 0
#可以將feature值轉成數字，但只會輸出一行
from sklearn.preprocessing import LabelEncoder
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(['male','female'])
    data['Sex'] = le.transform(data['Sex'])
    return data

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)

#Age Normalize
from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data['Age'] = scaler.fit_transform(data['Age'].values.reshape(-1,1))
    return data

train_data = normalize_age(train_data)
test_data = normalize_age(test_data)



#07 Split Data into X (features) & Y (Result) for later processing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"],axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction) #fraction from 1- 0.8

    return train_x.values, train_y,valid_x,valid_y

train_x,train_y,valid_x,valid_y = split_valid_test_data(train_data)

print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))
# .format為print的表達方式: http://www.runoob.com/python/att-string-format.html
print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))
# .shape輸出該矩陣的長寬數

#08

#Build Neural Network

from collections import namedtuple

def build_neural_network(hidden_units=10):

    #Setting
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None,1])
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.Variable(True, dtype=tf.bool)
#placeholder(tf.float32, shape=[None, train_x.shape[1]]) [x,y]-> x表示row數量,y表示column數
    
    #initial
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs,hidden_units, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)


    #Loss 
    logits = tf.layers.dense(fc, 1,activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    #Use AdamOptimizer Model
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    #Prediction and accuracy setting
    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    #Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training',
    'logits','cost','optimizer','predicted','accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


model = build_neural_network()


#09 Start modeling

def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y



epochs = 200  #Run 200 Times
train_collect = 50   
train_print=train_collect*2

learning_rate_value = 0.001
batch_size=16   #How big (single pieces ) you want do run at a time

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    for e in range(epochs):
        for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
            iteration+=1
            feed = {model.inputs: train_x,
                    model.labels: train_y,
                    model.learning_rate: learning_rate_value,
                    model.is_training: True
                }
            train_loss, _ ,train_acc = sess.run([model.cost,model.optimizer,model.accuracy],feed_dict=feed)

            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print==0:
                    print("Epoch: {}/{}".format(e+1,epochs),
                    "Train Loss: {:.4f}".format(train_loss),
                    "Train Acc: {:.4f}".format(train_acc))

                feed = {model.inputs: valid_x,
                        model.labels: valid_y,
                        model.is_training:False
                }

                #Sess run formula , nn model
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)

                #Print out result
                if iteration % train_print==0:
                    print("Espoch: {}/{}".format(e+1, epochs),
                        "Validation Loss: {:.4f}".format(val_loss),
                        "Validation Acc: {:.4f}".format(val_acc))

        saver.save(sess,"./titanic.ckpt")




