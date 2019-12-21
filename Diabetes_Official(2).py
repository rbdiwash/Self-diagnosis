
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import csv
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# In[10]:


pd.options.mode.chained_assignment = None
data2=pd.read_csv("C:/Users/pc/Desktop/5thSem/diabetes.csv")


# In[11]:


target_data2=data2["Outcome"]
del data2["Outcome"]
data2=data2.values
target_data2=target_data2.values
data2


# In[12]:


plt.plot(data2)


# In[5]:


model2=GaussianNB()
model2.fit(data2,target_data2)
expected2=target_data2
predicted2=model2.predict(data2)
confusion_matrix=metrics.confusion_matrix(expected2,predicted2)
accuracy2=(confusion_matrix[0][0]+confusion_matrix[1][1])/(len(data2))

print(accuracy2*100)
print(confusion_matrix)

