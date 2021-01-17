# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
 
keras.datasets.mnist.load_data()
(x_Train, y_Train), (x_Test, y_Test) = keras.datasets.mnist.load_data()


print('x_train_image:',x_Train.shape)
print('y_train_label:',y_Train.shape)

print('x_test_image:',x_Test.shape)
print('y_test_label:',y_Test.shape)

import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()
    

#x_Train[0]
#plot_image(x_Train[0])
#y_Train[0]

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
    
plot_images_labels_prediction(x_Train,y_Train,[],0,10)    
plot_images_labels_prediction(x_Test,y_Test,[],0,10)

#x_Train.shape
# 多加一個顏色的維度 
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

#x_Train4D.shape

# 將數值縮小到0~1
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

# 把類別做Onehot encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

#y_Trainy_Train
#y_TrainOneHot

#build the cnn model----------------------------------------------------------
print("start to build the model")
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()

#filter為16, Kernel size為(5,5),Padding為(same)#filter為 
#“valid”代表只进行有效的卷积，对边界数据不处理。“same”代表保留边界处的卷积结果
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))
# MaxPooling size為(2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Drop掉部分神經元避免overfitting
model.add(Dropout(0.25))


# 平坦化
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

print(model.summary())

#training model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize, 
                        y=y_TrainOneHot,validation_split=0.2, 
                        epochs=20, batch_size=300,verbose=2)


import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

#evaluate the performance-----------------------------------------------------
scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
print("evaluation")
print(scores[1])


#predict-----------------------------------------------------------------------
print("prediction figure")
prediction=model.predict_classes(x_Test4D_normalize)
print(prediction[:10])


def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)


#confusing matrix--------------------------------------------------------------
#print("confusing matrix")
#pd.crosstab(y_Test,prediction,
#            rownames=['label'],colnames=['predict'])
#
#df = pd.DataFrame({'label':y_Test, 'predict':prediction})
#print(df[(df.label==5)&(df.predict==3)])
#print(df[(df.label==5)&(df.predict==3)].index)
#
#
#df([x_Test[i] for i in df[(df.label==5)&(df.predict==3)].index],[y_Test[i] for i in df[(df.label==5)&(df.predict==3)].index],prediction,idx=0)
#
