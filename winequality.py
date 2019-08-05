# -*- coding: utf-8 -*-
"""
The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical measures of each wine.

It is a multi-class classification problem, but could also be framed as a regression problem. The number of observations for each class is not balanced. There are 4,898 observations with 11 input variables and one output variable. The variable names are as follows:

Fixed acidity.
Volatile acidity.
Citric acid.
Residual sugar.
Chlorides.
Free sulfur dioxide.
Total sulfur dioxide.
Density.
pH.
Sulphates.
Alcohol.
Quality (score between 0 and 10).
The baseline performance of predicting the mean value is an RMSE of approximately 0.148 quality points.
"""

import pandas as pd
import matplotlib.pyplot as mp

dataset=pd.read_csv('winequality-white.csv',sep=';')
X=dataset.iloc[:,0:11].values
Y=dataset.iloc[:,11].values

a=X[:,3]
mp.scatter(a,Y)