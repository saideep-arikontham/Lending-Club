#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('data/imputed_data.csv').set_index('id')
df


# ## Outlier analysis using SKLearn.

# In[2]:


#Lets try to use sklearn to detect outliers

from sklearn.ensemble import IsolationForest

outlier_percentage = 0.02

# Fit Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=outlier_percentage, random_state=42)
o1 = iso_forest.fit_predict(df)
o1


# In[3]:


from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(contamination=outlier_percentage)
o2 = lof.fit_predict(df)

o2


# In[4]:


#adding both series
import numpy as np

print(f'Outliers detected by each algorithms: {np.count_nonzero(o1 == -1)}')
outliers = o1 + o2
print(f'Outliers detected by both algorithms: {np.count_nonzero(outliers == -2)}')

df1 = df[outliers != -2] # == -2 because it means this has been identified as outlier by both the algorithms and removing only those


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Display the original and updated DataFrames
print(f'Before outlier detection : {df.shape[0]} rows')
print(f'After outlier detection : {df1.shape[0]} rows')


# In[6]:


df1['loan_status'].value_counts()


# In[7]:


df1.to_csv('data/removed_outlier.csv')


# In[ ]:




