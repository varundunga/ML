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
import matplotlib.pyplot as mp
#fetching dataset
'''url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
weburl=r.urlopen('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
rawdata=str(weburl.read())#reading the content from url



rawdata=rawdata.replace('\\n','\n')
data=rawdata[2:len(rawdata)-1]
data1=data.splitlines()
data2=np.array(data1)
data3=[]
for i in range(0,len(data2)):
    data3.append(data2[i].split(',')),

data4=np.array(data3)'''
#preprocessing the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
headers=[
        'Number of times pregnant',
        'Plasma glucose concentration',
        'Diastolic blood pressure (mm Hg)',
        'Triceps skinfold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)',
        'Diabetes pedigree function',
        'Age (years)',
        'Class variable (0 or 1)'
        ]
pima=pd.read_csv(url,header=None,names=headers)


#identifying input and output variables
    
Xs=pima.iloc[:,0:8].values
Y=pima.iloc[:,8].values
#X=Xs
#replacing missing values in input variables
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
#a=np.array([[0,1,2,3],[8,4,7,9]])

imputer = imputer.fit(Xs[:,0:8])
X = imputer.transform(Xs[:,0:8])

#Scaling the input variables

from sklearn.preprocessing import StandardScaler

X_SC=StandardScaler()
Xf=X_SC.fit_transform(X)

#splitting the dataset into train and test data sets

from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(Xf,Y,test_size=1/6,random_state=0)

#fitting dataset with classification models    

'''from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)'''

'''from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)'''

from sklearn.svm import SVC
classifier=SVC(random_state=0,kernel='rbf')
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)


'''from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)'''

'''from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=0)
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)'''

'''from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=5,random_state=0)
classifier.fit(X_Train,Y_Train)
Y_Pred=classifier.predict(X_Train)'''

#graphical representation

mp.plot()
#analysing predictions

from sklearn.metrics import confusion_matrix,accuracy_score
x=accuracy_score(Y_Test,classifier.predict(X_Test))
cm1=confusion_matrix(Y_Test,classifier.predict(X_Test))
(79+26)/(79+26+14+9)#82.03 for test set for logreg
(362+128)/(362+128+150)#76.5 for train set for logreg
(360+166)/(362+128+150)#82.18 for train set for KNN
(72+16)/(79+26+14+9)#68.75 for test set for KNN
(77+27)/(79+26+14+9)#81.25 for test set for SVC
(379+144)/(362+128+150)#81.71 for train set for SVC
(73+27)/(79+26+14+9)#78.125 for test set for NB
(343+135)/(362+128+150)#74.68 for train set for NB
(66+26)/(79+26+14+9)#71.875 for test set for Decision tree
(412+228)/(362+128+150)#100 for train set for Decision tree---Overfitting
(71+30)/(79+26+14+9)#78.9 for test set for random forest
(405+214)/(362+128+150)#96.71 for train set for random forest

df=pd.DataFrame({'x': range(1,11), 'y1': range(11,21), 'y2': range(21,31)})
 
# multiple line plot
mp.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
mp.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
mp.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
mp.legend()


