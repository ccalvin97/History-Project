#-*- coding:utf-8 -*-
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

train = pd.read_csv('train.csv',dtype={"Age": np.float64})
test = pd.read_csv('test.csv',dtype={"Age": np.float64})
PassengerId=test['PassengerId']

all_data = pd.concat([train, test], ignore_index = True)

##drop



#print(all_data["Fare"].isnull().any())
#qq=all_data["Fare"].dropna( axis=0,how='any').index
#qq=list(qq)
#all_data=all_data.iloc[qq,:]
#


print(all_data["Embarked"].isnull().any())
ww=all_data["Embarked"].dropna(axis=0,how='any').index
ww=list(ww)
all_data=all_data.iloc[ww,:]
all_data.reset_index(drop=True)

#--------------------------------------------------------------
#plt.figure(1) 
sns.barplot(x="Sex", y="Survived", data=train, palette='Set3')
print("Percentage of females who survived:%.2f" % (train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100))
print("Percentage of males who survived:%.2f" % (train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100))
#plt.show()

#plt.figure(2)
sns.barplot(x="Pclass", y="Survived", data=train, palette='Set3')
print("Percentage of Pclass = 1 who survived:%.2f" % (train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 2 who survived:%.2f" % (train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 3 who survived:%.2f" % (train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100))
#plt.show()

#plt.figure(3)
sns.barplot(x="SibSp", y="Survived", data=train, palette='Set3')
#plt.show()

#plt.figure(4)
sns.barplot(x="Parch", y="Survived", data=train, palette='Set3')
#plt.show()


facet1 = sns.FacetGrid(train, hue="Survived")
facet1.map(sns.kdeplot,'Age',shade= True)
facet1.set(xlim=(0, train['Age'].max()))
facet1.add_legend()



facet = sns.FacetGrid(train, hue="Survived")
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, 200))
facet.add_legend()


#plt.figure(7)
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')
#plt.show()

#plt.figure(8)
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data, palette='Set3')
#plt.show()

#plt.figure(9)
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data, palette='Set3')
#plt.show()

#plt.figure(10)
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data, palette='Set3')
#plt.show()

#plt.figure(11)
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')
#plt.show()
#plt.figure(12)
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')
#plt.show()


age_df = all_data[['Age', 'Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[ (all_data.Age.isnull()), 'Age' ] = predictedAges

all_data[all_data['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass",data=all_data, palette="Set3")



#all_data['Embarked']
#all_data['Embarked'] = all_data['Embarked'].fillna('C')

#all_data[all_data['Fare'].isnull()]

fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)


all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]

Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']


sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"], palette='Set3').set_xlabel('AverageSurvived')

Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']

sns.barplot(x=Male_Adult.index, y=Male_Adult['GroupCount'], palette='Set3').set_xlabel('AverageSurvived')

Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)

train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'



all_data=pd.concat([train, test])
aa=all_data
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
bb=all_data
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)


from sklearn.preprocessing import LabelBinarizer

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"],axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction) #fractionfrom 1- 0.8
    return train_x.values, train_y,valid_x,valid_y



train_x,train_y,valid_x,valid_y = split_valid_test_data(train)





#Build Neural Network----------------------------------------------------------



    #Setting

    
    # tf.layers.dense-> 构建了一个全连接网络

    #Loss 
    #get the max reaction and decide to the prediction class.

    #Use AdamOptimizer Model
#tensorflow的collection提供一个全局的存储机制，不会受到变量名生存空间的影响。一处保存，到处可取

    # tf.round函數-> 四捨五入
    
    #Export the nodes
    #參考資料
    #http://www.zlovezl.cn/articles/collections-in-python/
    #locals() 函数会以字典类型返回当前位置的全部局部变量



""" 參考:
https://zhuanlan.zhihu.com/p/25110150
"""

from collections import namedtuple

def build_neural_network(hidden_units=10):

    #Setting
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None,1])
    learning_rate = tf.placeholder(tf.float64)
    is_training = tf.Variable(True, dtype=tf.bool)

    
    #initial
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs,hidden_units, activation=None, kernel_initializer=initializer)
    fc = tf.layers.batch_normalization(fc, training=is_training)
    fc = tf.nn.relu(fc)


    #Loss 

    prediction = tf.layers.dense(fc, 2,activation=None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits=prediction)
    cost = tf.reduce_mean(cross_entropy)

    #Use AdamOptimizer Model
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#UPDATE_OPS is a collection of ops (operations performed when the graph runs, like multiplication, ReLU, etc.), not variables

    #Prediction and accuracy setting
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))  #get the max reaction and decide to the prediction class.
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))# cast 表示將原來的data轉換為其他type

#    predicted = tf.nn.sigmoid(logits)
#    correct_pred = tf.equal(tf.round(predicted), labels)
#    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    # tf.round函數-> 四捨五入
    
    #Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training',
    'prediction','cost','optimizer','accuracy']
    Graph = namedtuple('Graph', export_nodes)
    #參考資料
    #http://www.zlovezl.cn/articles/collections-in-python/
    local_dict = locals()
    #locals() 函数会以字典类型返回当前位置的全部局部变量

    graph = Graph1(*[local_dict[each] for each in export_nodes])

    return graph

model = build_neural_network()

#09 Start modeling-------------------------------------------------------------

def get_batch(data_x,data_y,batch_size):
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

    accuracy_valid = sess.run(model.accuracy,feed_dict={model.inputs:valid_x,model.labels:valid_y})
    accuracy_train = sess.run(model.accuracy,feed_dict={model.inputs:train_x,model.labels:train_y})           
    print('valid Acc: ' , accuracy_valid)
    print('Train Acc: ' , accuracy_train)


    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./my_model/titan_test_01_model")


print("start to test the data")
test=np.array(test, dtype=np.float64)   

#def get_test(test_x):
#    print("Start batchsizing")
#    for i in np.arange(test_x.shape[0]):
#        batch_x=test_x[i]
#        yield batch_x
#        
#    print(batch_x)
    

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./my_model/titan_test_01_model")
    
     
    #    for epoch_test in get_test(test):        
    aaa=sess.run(model.prediction, feed_dict= {model.inputs:test})    
    print("prediction是", model.prediction.eval({model.inputs:test}))
        
    result = tf.argmax(model.prediction,1)
    print("result:", result.eval({model.inputs:test}))
    img_numpy=result.eval({model.inputs:test })
    #    print("out2=",type(img_numpy))
    #    
    #    
    #    submission=pd.DataFrame(img_numpy)
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": img_numpy.astype(np.int64)})
    submission.to_csv("submission_titan_測試版.csv", index=False)



