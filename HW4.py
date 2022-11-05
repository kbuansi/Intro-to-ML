#!/usr/bin/env python
# coding: utf-8

# In[44]:


import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from ipywidgets import interact, fixed
from sklearn.datasets import make_circles
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from scipy import stats
import seaborn as sns; sns.set()


# In[33]:


cancerData = load_breast_cancer()

cancerX = pd.DataFrame(cancerData.data)

cancerY = pd.DataFrame(cancerData.target)


sc = StandardScaler()

cancerXSTD = sc.fit_transform(cancerX)

cancerXSTD


# In[46]:


N= 25
for i in range(1,25):
    
    print('principle component(s)=' ,i)
        
    pca = PCA(n_components = i)
    
    principalComponents = pca.fit_transform(cancerXSTD)
        
    principalDF = pd.DataFrame(data = principalComponents)
    
    principalDF
    
    xTrain,xTest,yTrain,yTest = train_test_split(principalDF,cancerY,test_size = .2,random_state=42)  
    
    C = [.1]
    
    for c in C:
        
        clf = LogisticRegression(penalty = 'l1', C=c, solver = 'liblinear')
        
        clf.fit(xTrain, yTrain)
        
        print('c:', c)
        
        print( 'Training accuracy:', clf.score(xTrain,yTrain))
        
        print('test accuracy:', clf.score(xTest,yTest))
        
        print('')
        
    warnings.filterwarnings('ignore')
    
    predicted = clf.predict(xTest)
    
    matrix = confusion_matrix(yTest,predicted)
    
    print(matrix)
 
    report = classification_report(yTest,predicted)
    
    print(report)
        


# In[32]:


pca = PCA(n_components = 5)

principalComponents = pca.fit_transform(cancerXSTD)

principalDF = pd.DataFrame(data = principalComponents)


# In[23]:


clf = svm.SVC(kernel = 'linear',C=1E6)

clf.fit(xTrain,yTrain)

predictedSVM = clf.predict(xTest)

cfLinear = confusion_matrix(yTest,predictedSVM)

reportLinear = classification_report(yTest,predictedSVM)

print(cfLinear)

print(reportLinear)

print('test accuracy', metrics.accuracy_score(yTest,predictedSVM))


# In[24]:


clf = svm.SVC(kernel = 'rbf',C=1E6)

clf.fit(xTrain,yTrain)

predictedSVM = clf.predict(xTest)

cfLinear = confusion_matrix(yTest,predictedSVM)

reportLinear = classification_report(yTest,predictedSVM)

print(cfLinear)

print(reportLinear)

print('test accuracy', metrics.accuracy_score(yTest,predictedSVM))


# In[25]:


clf = svm.SVC(kernel = 'sigmoid',C=1E6)

clf.fit(xTrain,yTrain)

predictedSVM = clf.predict(xTest)

cfLinear = confusion_matrix(yTest,predictedSVM)

reportLinear = classification_report(yTest,predictedSVM)

print(cfLinear)

print(reportLinear)

print('test accuracy', metrics.accuracy_score(yTest,predictedSVM))


# In[35]:


pca = PCA(n_components = 4)

principalComponents = pca.fit_transform(cancerXSTD)

principalDF = pd.DataFrame(data = principalComponents)

xTrain,xTest,yTrain,yTest = train_test_split(principalDF,cancerY,test_size = .2,random_state=42)

clf = svm.SVC(kernel = 'linear')

clf.fit(xTrain,yTrain)

predictedSVM = clf.predict(xTest)

cfLinear = confusion_matrix(yTest,predictedSVM)

reportLinear = classification_report(yTest,predictedSVM)

print(cfLinear)

print(reportLinear)

print('test accuracy', metrics.accuracy_score(yTest,predictedSVM))


# In[43]:


X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_lin = SVR(kernel='linear', C=1e3)

svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X, y).predict(X)

y_lin = svr_lin.fit(X, y).predict(X)

y_poly = svr_poly.fit(X, y).predict(X)

lw = 2

plt.scatter(X, y, color='darkorange', label='data')

plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')

plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')

plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')

plt.xlabel('data')

plt.ylabel('target')

plt.title('SUpport Vector Regression')

plt.legend()

plt.show()





# In[ ]:





# In[ ]:





# In[ ]:




