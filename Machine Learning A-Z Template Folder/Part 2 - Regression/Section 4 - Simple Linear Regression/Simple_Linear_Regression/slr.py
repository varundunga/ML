#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

#importing the dataset

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values  
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
'''z=dataset.iloc[:,:].values  
a=len(z[0])'''

# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
mp.scatter(X_train, y_train, color = 'red')
mp.plot(X_train, regressor.predict(X_train), color = 'blue')
mp.title('Salary vs Experience (Training set)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()

# Visualising the Test set results
mp.scatter(X_test, y_test, color = 'red')
mp.plot(X_train, regressor.predict(X_train), color = 'blue')
mp.title('Salary vs Experience (Test set)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()


