import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
dataset=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')
data=dataset[['Complaint Type','City']]
#data=data.iloc[0:100,:]
FinalData=[]
ucity=data['City'].unique()
ucomplaintdata=data['Complaint Type'].unique()

for ctype in ucomplaintdata:
    Complaint=[]
    City=[]
    Count=[]
    Complaint.append(ctype)
    for city in ucity:
        City.append(city)
        Count.append(data['Complaint Type'][data['Complaint Type'] == ctype][data['City'] == city].count())
    Complaint.append(City)
    Complaint.append(Count)
    FinalData.append(Complaint)
Complaint=[]
City=[]
Count=[]
City=FinalData[0][1]
for i in range(0,len(FinalData)):
    Complaint.append(FinalData[i][0])
    Count.append(FinalData[i][2])
    
XIndex=np.arange(len(City))
k=len(City)*[0]
for i in range(0,len(Count)):
    
    if(i==0):
        plt.bar(XIndex,Count[i],edgecolor='Black')
    else:
        plt.bar(XIndex,Count[i],bottom=k,edgecolor='Black')
    k=np.add(k,Count[i])
    for j in XIndex:
        if((Count[i][j])!=0):
            plt.text(j,k[j]-((Count[i][j])/2),(Count[i][j]),size=3)
    
plt.xlabel('City')  
plt.ylabel('Complaint count')
#plt.ylim(0,max(k))  
plt.xticks(XIndex,City,fontsize=5,rotation=90)
plt.legend(Complaint,fontsize=5,loc='upper right')
plt.title('City vs Complaint Count')
plt.grid()
#plt.figure(figsize=(2,1))
plt.savefig('final1.png')

    
