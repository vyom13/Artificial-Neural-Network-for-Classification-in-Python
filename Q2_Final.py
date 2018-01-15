
# coding: utf-8

# In[9]:



from __future__ import division
from sklearn import datasets
import pandas as pd
import numpy as np


iris = datasets.load_iris()
df = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1), columns=iris.feature_names + ['target'])

df1 = df.drop(df.index[0:50])
df2 = df1.drop(['sepal length (cm)','sepal width (cm)'],axis = 1)



target = ['1', '2']

m = df2.shape[0]

n = 2  #features

k = 2 #Number of classes

X = np.ones((m,n + 1))
y = np.array((m,1))
X[:,1] = df2['petal length (cm)'].values
X[:,2] = df2['petal width (cm)'].values


y = df2['target'].values
y=y-1
#TO ignore the warnings given while scaling.
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

#Scaling the values between 0 and 1.
for j in range(1,3):
    X[:, j] = (X[:, j] - min(X[:,j]))/(max(X[:,j]) - min(X[:,j]))

#Activation Function
def sigmoid(z):
    return (1/(1+np.exp(-z)))

#Creating the first weight matrix with random values from 0 to 1.
th1 = np.random.random((2,3))
#Creating the second weight matrix with random values from 0 to 1.
th2 = np.random.random((1,3))

#function to calculate the a2 matrix
def calculate_a2(X,th1):
    z2 = np.dot(X,th1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones([len(a2),1]),a2,1)
    return a2

#function for predicting a2 
def calculate_a2_prediction(X_test,th1):
    z2 = np.dot(X_test.reshape(1,3),th1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones([len(a2),1]),a2,1)
    return a2

#function for calculating a3
def calculate_a3(a2,th2):
    z3 = np.dot(a2,th2.T)
    a3 = sigmoid(z3)
    return a3
    

def CostFunction(th2, a2, y):
    m = len(y)
    a3 = calculate_a3(a2,th2)
    cost = (-1/m) * (y.T.dot(np.log(a3)) + (1 - y).T.dot(np.log(1 - a3)))
    return cost

def gradient_th2(th2,a2,X,y):
    m,n = X.shape
    a3 = calculate_a3(a2,th2)
    delta = a3 - y.reshape(m,1)   
    gradient = np.dot(np.transpose(delta),X)
    gradient  = gradient/m
    return gradient


def gradient_th1(th1,th2,a2,X,y):
    m,n = X.shape
    a2 = calculate_a2(X,th1)
    a3 = calculate_a3(a2,th2)
    delta2 = (a3 - y.reshape(m,1))
    delta1 = np.multiply(np.dot(delta2,th2),np.multiply(a2,(1-a2)))
    gradient = np.dot(np.transpose(delta1),X)
    gradient = gradient[1:3,:]
    gradient = gradient/m
    return gradient    
    


def predictions(th1,th2,X):
    a2 = calculate_a2_prediction(X,th1)
    a3_predict = calculate_a3(th2,a2)
    if (a3_predict >= 0.5):
        return 1
    else:
        return 0 


total_error = 0

    
for k in range(100):
    X_test = X[k,:]
    X_train = np.delete(X,k,axis=0)
    Y_train = np.delete(y,k)
    Y_test = y[k]
    th1 = np.random.random((2,3))
    th2 = np.random.random((1,3))
    list_cost = []
 

       
    for i in range(1000):
        learning_rate = 0.5
        a2 = calculate_a2(X_train,th1)
        a3 = calculate_a3(a2,th2)
        cost = CostFunction(th2,a2,Y_train)
        gradient2 = gradient_th2(th2,a2,X_train,Y_train)
        gradient1 = gradient_th1(th1,th2,a2,X_train,Y_train)
        th2 -= np.multiply(learning_rate,gradient2)
        th1 -= np.multiply(learning_rate,gradient1)
        prediction = predictions(th1,th2,X_test)
        error_each_iter = abs(Y_test - prediction)
    total_error = total_error + error_each_iter
Average_Error_Rate =     total_error/100.0
print('Average_Error_Rate',Average_Error_Rate)



   
    


# In[2]:




# In[3]:


    
    
    


# In[4]:




# In[ ]:




# In[ ]:



