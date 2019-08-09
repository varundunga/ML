import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#url='https://data.consumerfinance.gov/api/views.json'
dataset=pd.read_csv('Consumer_Complaints1.csv',encoding='unicode_escape')
dataset1=dataset[['Product', 'Consumer complaint narrative']]
df=dataset1
df = df[pd.notna(df['Consumer complaint narrative'])]
df=df.iloc[0:10,:]
df['category_id'] = df['Product'].factorize()[0]
fig = plt.figure(figsize=(8,6))
df.groupby('Product').count().plot.bar(ylim=0)
plt.show()
k=df.groupby('Product').count().iloc[:,0]
k.plot.bar(ylim=0)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['Consumer complaint narrative']).toarray()

'''x=['asdf adf adse ad sadf aadsf ads','adsf ga a ae g a','asdf adf adse ad sadf aadsf ads','adsf ga a ae g a''asdf adf adse ad sadf aadsf ads','adsf ga a ae g a','good a a a a a a abad','worse a a a a good bad','asdf adf adse ad sadf aadsf ads','adsf ga a ae g a','asdf adf adse ad sadf aadsf ads','adsf ga a ae g a''asdf adf adse ad sadf aadsf ads','adsf ga a ae g a','good a a a a a a abad','worse a a a a good bad']

y=pd.DataFrame(x,columns=['d'])
y=y.iloc[0:9,:]
features=tfidf.fit_transform(y['d']).toarray()'''

from sklearn.feature_selection import chi2
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)

N = 2
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
