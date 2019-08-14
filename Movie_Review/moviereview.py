import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
dataset=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')
data=dataset[['Complaint Type','City']]