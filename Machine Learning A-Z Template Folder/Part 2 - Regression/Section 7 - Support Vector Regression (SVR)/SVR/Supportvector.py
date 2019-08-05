

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.svm import SVR

#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

#Data preprocessing-Feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Sc_y=StandardScaler()
X=Sc_X.fit_transform(X)
y=Sc_y.fit_transform(y)


#fitting with SVR model
regressor=SVR()
regressor.fit(X,y)


y_pred=regressor.predict([[6.5]])
y_pred=Sc_y.inverse_transform(y_pred)
pp.scatter(X,y,color='blue')
pp.plot(X,regressor.predict(X),color='red')

#increasing the resolution

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

pp.scatter(X,y,color='blue')
pp.plot(X_grid,regressor.predict(X_grid),color='red')
#plt.switch_backend('TkAgg')
#mng=plt.get_current_fig_manager()
#mng.window.showMaximized()
pp.show()

