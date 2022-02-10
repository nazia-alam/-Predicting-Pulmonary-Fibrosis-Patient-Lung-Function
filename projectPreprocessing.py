#!/usr/bin/env python
# coding: utf-8

# In[1]:

# This program performs some preprocessing on the data before applying the prediction functions

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import keras
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


# In[2]:


df = pd.read_csv("train.csv")
print(df.shape)
print(df)


# In[3]:


# perform min_max normalization of numeric feature values

scaler = MinMaxScaler()
#df_numeric = pd.DataFrame()
#df_numeric = df.iloc[:,1:5]
#print(df_numeric)
df_v2 = df
df_v2.iloc[:,2:5] = scaler.fit_transform(df.iloc[:,2:5])
print(df_v2)
print(df_v2.shape)


# In[4]:


#df_sex = np_utils.to_categorical(df_v2['Sex'], 2)
#df_smoke = np_utils.to_categorical(df_v2['SmokingStatus'], 3)
df_sex = {}
df_smoke = {}
encoder = OneHotEncoder()
df_sex = encoder.fit_transform(df_v2.iloc[:,5:6])
print(df_sex.shape)
#print(df_sex)
df_smoke = (df['SmokingStatus']).to_numpy()
print(df_smoke.shape)

df_smoke = df_smoke.reshape(1549,1)
print(df_smoke.shape)
df_smoke = encoder.fit_transform(df_smoke)
print(df_smoke.shape)
print(df_smoke[0,:])


# In[5]:


X = df_v2.iloc[:,5:7]


#le = preprocessing.LabelEncoder()
#X2 = X.apply(le.fit_transform)
#print(X2.shape)

enc = preprocessing.OneHotEncoder()

enc.fit(X)

onehotlabels = enc.transform(X).toarray()
onehotlabels.shape
print(onehotlabels)
df_v2['sex_f'] = onehotlabels[:,0]
df_v2['sex_m'] = onehotlabels[:,1]
df_v2['smoke_current'] = onehotlabels[:,2]
df_v2['smoke_ex'] = onehotlabels[:,3]
df_v2['smoke_never'] = onehotlabels[:,4]
print(df_v2.shape)
df_v2 = df_v2.drop(['Sex','SmokingStatus'],axis=1)
print(df_v2.shape)
print(df_v2.head())


# In[6]:


#np.savetxt('train_preprocessed.csv',df_v2,delimiter=',')

df_v2.to_csv("train_preprocessed.csv")


# In[ ]:




