import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint

dataFrame = pd.read_csv('train.csv',header=0)
dataFrame['Gender']  = dataFrame['Sex'].map({'female':0,'male':1}).astype(int)
medianAges = np.zeros((2,3))
for i in range (0,2):
    for j in range(0,3):
        medianAges[i,j]= dataFrame[(dataFrame['Gender'] == i) &(dataFrame['Pclass']==j+1)]['Age'].dropna().median()
dataFrame['AgeFill'] = dataFrame['Age']
for i in range(0,2):
    for j in range(0,3):
        dataFrame.loc[(dataFrame.Age.isnull())&(dataFrame['Gender']==i)&(dataFrame['Pclass']==j+1),'AgeFill'] =medianAges[i,j]
dataFrame['familySize'] = dataFrame['SibSp']+dataFrame['Parch']

dataFrame=dataFrame.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','PassengerId','SibSp','Parch'], axis=1)
dataFrame=dataFrame.dropna()
dataFrame.insert(0,1,1)
Y = dataFrame['Survived'].values
for i in range(0,Y.size):
    if Y[i] == 0 :
        Y[i] = -1
    else :
        Y[i] = 1
# print(Y)
dataFrame=dataFrame.drop(['Survived'],axis=1)
# print(dataFrame.head())
################## Linear Regression ############
X = dataFrame.values
# print(dataFrame.head())
xT = X.transpose()
xTx = np.dot(xT,X)
xTxInv = np.linalg.inv(xTx)
xD = np.dot(xTxInv,xT)
# print(np.shape(xD))
W = np.dot(xD,Y)
# print(np.shape(Y))
# print(W)

###################Perceptron Algorithm#################
# n = 1000
# minError = 9999999999
# minW = W
# for i in range(0,n):
#     r = randint(0,len(X)-1)
#     xN = X[r]
#     result = np.dot(W,xN)
#     yN = Y[r]
#     error = yN - result
#     # print(result)
#     if(minError > error):
#         minError = error
#         minW = W
#     W = W + np.dot(yN,xN)
# print(minW)

#####################Test Data##########################
dataFrame = pd.read_csv('test.csv',header=0)
dataFrame['Gender']  = dataFrame['Sex'].map({'female':0,'male':1}).astype(int)
medianAges = np.zeros((2,3))
for i in range (0,2):
    for j in range(0,3):
        medianAges[i,j]= dataFrame[(dataFrame['Gender'] == i) &(dataFrame['Pclass']==j+1)]['Age'].dropna().median()
dataFrame['AgeFill'] = dataFrame['Age']
for i in range(0,2):
    for j in range(0,3):
        dataFrame.loc[(dataFrame.Age.isnull())&(dataFrame['Gender']==i)&(dataFrame['Pclass']==j+1),'AgeFill'] =medianAges[i,j]
dataFrame['familySize'] = dataFrame['SibSp']+dataFrame['Parch']
pID = dataFrame['PassengerId'].values
dataFrame=dataFrame.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','PassengerId','SibSp','Parch'], axis=1)
dataFrame.insert(0,1,1)
X = dataFrame.values
results = []
for i in range(0,len(X)):
    results.append(np.dot(W,X[i]))
for i in range(0,len(results)):
    if(results[i]>0):
        results[i] = 1
    else:
        results[i] = 0
submission = pd.DataFrame({
        "PassengerId": pID,
        "Survived": results
    })
submission.to_csv("perceptron.csv", index=False)