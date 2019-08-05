# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:03:34 2019

@author: vdunga
"""
#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix





#importing DATASET
DATASET = pd.read_csv('Social_Network_Ads.csv')
XS = DATASET.iloc[:, 2:4].values
Y = DATASET.iloc[:, 4].values
#l=XS[Y == 0, 0]

#Y=Y.reshape(400,1)
#feature scaling

X_SC = StandardScaler()
#Y_Sc=StandardScaler()
X = X_SC.fit_transform(XS)
#Y=Y_Sc.fit_transform(Y)

#splitting DATASET into TRAIN and TEST sets


X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=1/4, random_state=0)
#Y_TRAIN1=Y_TRAIN.ravel().T

#fitting logistic regression CLASSIFIER to DATASET

CLASSIFIER = KNeighborsClassifier()
CLASSIFIER.fit(X_TRAIN, Y_TRAIN)

#predicting values

Y_PRED = CLASSIFIER.predict(X_TEST)

#Making confusion matrix to know count of correct predictions

CM = confusion_matrix(Y_TEST, Y_PRED)

#visualizing the predictions vs actual

X_SET, Y_SET = X_TRAIN, Y_TRAIN
'''xx,yy=np.meshgrid(XS[:,0],XS[:,1])
xx=xx*1000
mp.contourf(xx,yy,yy,cmap = ListedColormap(('red', 'green','blue')))'''
X1, X2 = np.meshgrid(np.arange(start=min(X_SET[:, 0])-1, stop=max(X_SET[:, 0])+1, step=0.01),
                     np.arange(start=min(X_SET[:, 1])-1, stop=max(X_SET[:, 1])+1, step=0.01))
mp.xlim(X1.min(), X1.max())
mp.ylim(X2.min(), X2.max())
mp.contourf(X1, X2, CLASSIFIER.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(Y_SET)):
    mp.scatter(X_SET[Y_SET == j, 0], X_SET[Y_SET == j, 1],
               c=ListedColormap(('red', 'green'))(i), label=j)
mp.title('Logistic Regression (TEST set)')
mp.xlabel('Age')
mp.ylabel('Estimated Salary')
mp.legend()
mp.show()

'''print(list(enumerate(np.unique(Y_SET))))
a=np.array([X1.ravel(),X2.ravel()]).T
r=X1.shape
v = np.arange(1, 10)
w = v.reshape(-1, 1)'''
