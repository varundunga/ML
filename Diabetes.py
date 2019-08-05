'''3. Pima Indians Diabetes Dataset
The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.

It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. Missing values are believed to be encoded with zero values. The variable names are as follows:

Number of times pregnant.
Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
Diastolic blood pressure (mm Hg).
Triceps skinfold thickness (mm).
2-Hour serum insulin (mu U/ml).
Body mass index (weight in kg/(height in m)^2).
Diabetes pedigree function.
Age (years).
Class variable (0 or 1).
The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 65%. Top results achieve a classification accuracy of approximately 77%.'''


import urllib.request as r
import pandas as pd
import numpy as np
#fetching dataset

weburl=r.urlopen('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
rawdata=str(weburl.read())#reading the content from url

#preprocessing the dataset

rawdata=rawdata.replace('\\n','\n')
data=rawdata[2:len(rawdata)-1]
data1=data.splitlines()
data2=np.array(data1)
data3=[]
for i in range(0,len(data2)):
    data3.append(data2[i].split(','))

data4=np.array(data3)

#identifying input and output variables
    
X=data4[:,0:8]
Y=data4[:,8]

#Scaling the input variables

from sklearn.preprocessing import StandardScaler

X_SC=StandardScaler()
Xs=X_SC.fit_transform(X)

#splitting the dataset into train and test data sets

from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(Xs,Y,test_size=1/6,random_state=0)

#fitting dataset with classification models    

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)

#analysing predictions

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_Test,classifier.predict(X_Test))


