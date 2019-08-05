

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting with linear regression model

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X,y)

y_pred=lin_reg.predict(X)

pp.scatter(X,y,color='blue')
pp.plot(X,lin_reg.predict(X),color='red')

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

pp.scatter(X,y,color='blue')
pp.plot(X_grid,lin_reg.predict(X_grid),color='red')

#fitting with polynomial regression model

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_poly=LinearRegression()
lin_poly.fit(X_poly,y)

pp.scatter(X,y,color='blue')
pp.plot(X_grid,lin_poly.predict(poly_reg.fit_transform(X_grid)),color='red')
pp.show()
