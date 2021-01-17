import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train_pm2.5.csv', 'r', encoding='big5') 
row = csv.reader(text)
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
               data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = 2*np.dot(x_t,loss)
    s_gra += gra**2                 #ok
    ada = np.sqrt(s_gra)            #ok
    w = w - l_rate * gra/ada        ##更新learning rate
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
    
# save model
np.save('model.npy',w)
# read model
w = np.load('model.npy')

test_x = []
n_row = 0
text = open('test_pm2.5.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "resultpredict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

##檢查正確率
win=0
fail=0
ratio=0

res = open('resultpredict.csv', 'r', encoding='big5')
ans = open('ans_pm2.5.csv', 'r', encoding='big5')

row1 = csv.reader(res)
row2 = csv.reader(ans)

personDF1 = []
for i in row1:
    personDF1.append(i)
personDF2 = []
for i in row2:
    personDF2.append(i)    
    
r3=list(map(lambda x,y: x-y,zip(personDF1, personDF2)))

    
for i1 in range(239):
    if -1 <= r3[i1] <= 1:
        win=win+1  
    else: 
        fail=fail+1
            
res.close()
ans.close()

ratio=(win/ (win+fail))
print(ratio)



