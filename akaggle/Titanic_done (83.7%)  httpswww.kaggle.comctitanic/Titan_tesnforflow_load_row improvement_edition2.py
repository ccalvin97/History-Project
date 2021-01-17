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


#Drop train_data
#train=train.drop('Title_Master',axis=1)
#train=train.drop('Title_Miss',axis=1)
#train=train.drop('Title_Mr',axis=1)
#train=train.drop('Title_Mrs',axis=1)

train_x = train.as_matrix()[:,1:]
train_y = train.as_matrix()[:,0]

#split
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.3)

##將y轉換成 2 columns
# 注意: y 的不同類別要分成不同column表示

def column_change(x, column_count):
    x_shape=x.shape[0]
    xx=np.zeros((x_shape,column_count))
    for i in np.arange(x_shape):
        if x[i]==1:
            xx[i,1]=1
    return xx

train_y=column_change(train_y, 2)
valid_y=column_change(valid_y, 2)          


#data處理完畢------------------------------------------------------
#進入module--------------------------------------------------------
#------------------------------------------------------------------
n_nodes_hy1 = 300
n_nodes_hy2 = 1000
n_nodes_hy3 = 50
n_nodes_hy4 = 100
n_nodes_hy5 = 40
n_nodes_hy6 = 20
n_classes = 2
#fit with one-hot format
# number = 1 [ 0,1,0,0,0,0,0,0,0,0,0,0]


batch_size =16 #run time , piece by piece


#setting format
#x = tf.placeholder('float',[None, 25],name="my_x")
x = tf.placeholder('float',[None, 25],name="my_x")
y = tf.placeholder('float',name="my_y")


#Setting format for layer's w,b
#下一行中[x,y]-> x表示layer輸入node數, y表示layer輸出node數 
hy_1_layer = {'weights':tf.Variable(tf.random_normal([25,n_nodes_hy1])),'biases':tf.Variable(tf.random_normal([n_nodes_hy1]))}
hy_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy1,n_nodes_hy2])),'biases':tf.Variable(tf.random_normal([n_nodes_hy2]))}
hy_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy2,n_nodes_hy3])),'biases':tf.Variable(tf.random_normal([n_nodes_hy3]))}
hy_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy3,n_nodes_hy4])),'biases':tf.Variable(tf.random_normal([n_nodes_hy4]))}
hy_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy4,n_nodes_hy5])),'biases':tf.Variable(tf.random_normal([n_nodes_hy5]))}
hy_6_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy5,n_nodes_hy6])),'biases':tf.Variable(tf.random_normal([n_nodes_hy6]))}
output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hy6,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

#Layout the structure here!
layer_1 = tf.add(tf.matmul(x,hy_1_layer['weights']), hy_1_layer['biases'])
layer_1 = tf.nn.relu(layer_1)
#用 RELU 替换你所有的 sigmoid，然后你会得到一个更快的初始收敛，李宏毅

layer_2 = tf.add(tf.matmul(layer_1,hy_2_layer['weights']), hy_2_layer['biases'])
layer_2 = tf.nn.relu(layer_2)

layer_3 = tf.add(tf.matmul(layer_2,hy_3_layer['weights']), hy_3_layer['biases'])
layer_3 = tf.nn.relu(layer_3)

layer_4 = tf.add(tf.matmul(layer_3,hy_4_layer['weights']), hy_4_layer['biases'])
layer_4 = tf.nn.relu(layer_4)

layer_5 = tf.add(tf.matmul(layer_4,hy_5_layer['weights']), hy_5_layer['biases'])
layer_5 = tf.nn.relu(layer_5)

layer_6 = tf.add(tf.matmul(layer_5,hy_6_layer['weights']), hy_6_layer['biases'])
layer_6 = tf.nn.relu(layer_6)


prediction = tf.matmul(layer_6,output_layer['weights']) + output_layer['biases']

#def train_NN(x):
#Call sub-function,  Create the model called NN_model()

#Setting training parameters here!
#Calculate the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
#Try to tune the parameter (w,b) to reduce the cost
optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost)
#AdamOptimizer 李宏毅

#softmax_cross_entropy_with_logits包含两个作用：1、计算softmax，2、求cross_entropy。     
#      除去name参数用以指定该操作的name，与方法有关的一共两个参数： 
#      第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes 
#      第二个参数labels：实际的标签，大小同上。 
#      第一步是先对网络最后一层的输出做一个softmax，
#      第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵，公式如下：
#      最后求一个平均，得到我们想要的loss 
#      logits放模型的output

#def next_batch(num, data, labels):
#    '''
#    Return a total of `num` random samples and labels. 
#    '''
#    idx = np.arange(len(data))
#    np.random.shuffle(idx)
#    idx = idx[:num]
#    data_shuffle = [data[ i] for i in idx]
#    labels_shuffle = [labels[ i] for i in idx]
#
#    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_batch(data_x,data_y,batch_size):
#    print("Start batchsizing")
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        yield batch_x,batch_y

#
#train_y = train_y.reshape(train_y.shape[0],2)
#valid_y = valid_y.reshape(valid_y.shape[0],2)

hm_epochs = 2200
#hm_epochs = 200 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for epoch_x, epoch_y in get_batch(train_x, train_y, batch_size):
        
   #         print("epoch_x",epoch_x)
    #        print("epoch_y",epoch_y)

#        for _ in range(int(len(train_x)/batch_size)):          
#            train_x = tf.cast(train_x, tf.float32)
#            train_y = tf.cast(train_y, tf.int32)
#            input_queue = tf.train.slice_input_producer([train_x, train_y], shuffle=False,num_epochs=25)
#            epoch_x, epoch_y = tf.train.batch(input_queue, batch_size)
##            epoch_x=epoch_x.eval()
##            epoch_y=epoch_y.eval()
#            epoch_x,epoch_y = get_batch( train_x, train_y, batch_size)
        
            _, c = sess.run([optimizer,cost],feed_dict={x: epoch_x, y: epoch_y})

            epoch_loss += c
        if epoch % 10==0:
            print('Epoch', epoch, 'competed out of ', hm_epochs, 'loss: ', epoch_loss)
        
    #prediction=tf.cast(prediction, tf.float64)
    #y=tf.cast(y, tf.float64)
    #Reshape y form [889,]  to [889,1]
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))  #get the max reaction and decide to the prediction class.
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))# cast 表示將原來的data轉換為其他type
# argmax 是輸出
#---------------------------------
#    cc=correct.eval({x: train_x, y: train_y})
#    print(cc)
    
    #Change input from train_x to valid_x,  
    print('Accuracy: ' , accuracy.eval({x: valid_x, y: valid_y}))
    #Save the session here
    oSaver = tf.train.Saver()
    oSess = sess
    oSaver.save(oSess,"./my_model/titan_test_01_model")
    
# start for testing set-------------------  
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
    
 
#    for epoch_test in get_test(tesfile:///C:/Users/ccalvin97/.spyder-py3/akaggle/Titan/Titan_tesnforflow_load_row improvement_edition2_66%25_ok.pyt):        
    aaa=sess.run(prediction, feed_dict= {x:test})    
    print("prediction是", prediction.eval({x:test}))
        
    result = tf.argmax(prediction,1)
    print("result:", result.eval({x:test}))
    img_numpy=result.eval({x:test })
#    print("out2=",type(img_numpy))
#    
#    
#    submission=pd.DataFrame(img_numpy)
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": img_numpy.astype(np.int32)})
    submission.to_csv("submission.csv", index=False)
#-----------------------------------------------------------
    
#predictions = pipeline.predict(test)
#submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
#submission.to_csv("submission.csv", index=False)
