# -*- coding: utf-8 -*-
"""

The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.

It is a regression problem. It is comprised of 63 observations with 1 input variable and one output variable. The variable names are as follows:

Number of claims.
Total payment for all claims in thousands of Swedish Kronor.
The baseline performance of predicting the mean value is an RMSE of approximately 81 thousand Kronor.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request as r

weburl=r.urlopen("https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt")
rawdata=str(weburl.read())
rawdata1=rawdata.splitlines()

rawdata2=rawdata[rawdata.index('\\nX\\t'):]
rawdata3=rawdata2.replace('\\n','\n').replace('\\t','\t').replace(',','.')
rawdata4=rawdata3[1:len(rawdata3)-1]
rawdata5=rawdata4.splitlines()
rawdata6=[]
for i in range(0,len(rawdata5)):
    rawdata6.append(rawdata5[i].split())
    
rawdata7=np.array(rawdata6)
url='https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt'
insr=pd.read_csv(url)

