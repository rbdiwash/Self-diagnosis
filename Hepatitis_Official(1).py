
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[2]:


pd.options.mode.chained_assignment = None
data=pd.read_csv("C:/Users/pc/Desktop/5thSem/hepatitis_csv.csv")
data=data.replace(r'\s+', np.nan, regex=True)


# In[3]:




# True or False Transformation
def transform(x):
    return 1 if x else 0

# Mean Value Fill
def  Mean_ValFill(col_name,value):
    data[col_name].fillna(value,inplace=True)
    
# Scaling Data for equal weightage
def Scaling_data(dataset,col_name):
   
    Max_Val=max(dataset[col_name])
    Min_Val=min(dataset[col_name])
    
    for i in range(len(dataset)):
        dataset[col_name][i]=(dataset[col_name][i]-Min_Val)/(Max_Val-Min_Val)
        dataset[col_name][i]="{0:.4f}".format(dataset[col_name][i])
        
    return dataset[col_name]

# Sex Transformation
def trans_sex(x):
    if x=='male':
        return 1
    if x=='female':
        return 0
    
# Class Transformation  
def trans_class(x):
    if x=='live':
        return 1
    if x=='die':
        return 0
    


  
data["steroid"]=data["steroid"].apply(transform)
data["antivirals"]=data["antivirals"].apply(transform)
data["fatigue"]=data["fatigue"].apply(transform)
data["malaise"]=data["malaise"].apply(transform)
data["anorexia"]=data["anorexia"].apply(transform)
data["liver_big"]=data["liver_big"].apply(transform)
data["liver_firm"]=data["liver_firm"].apply(transform)
data["spleen_palpable"]=data["spleen_palpable"].apply(transform)
data["spiders"]=data["spiders"].apply(transform)
data["ascites"]=data["ascites"].apply(transform)
data["varices"]=data["varices"].apply(transform)
data["histology"]=data["histology"].apply(transform)




# In[4]:


# Mean Value Fill
Mean_ValFill("albumin",4.25)
Mean_ValFill("alk_phosphate",80)
Mean_ValFill("sgot",22.5)
Mean_ValFill("protime",12)

# Class Transformation
data["class"]=data["class"].apply(trans_class)

# Sex Transformation 
data["sex"]=data["sex"].apply(trans_sex)


# In[5]:



# Hepatitis Dataset    
# Bilirubin_Scaled=Scaling_data(data,"bilirubin")
# Alkaline_Scaled=Scaling_data(data,"alk_phosphate")
# Sgot_Scaled=Scaling_data(data,"sgot")
# Protime_Scaled=Scaling_data(data,"protime")
# Albumin_Scaled=Scaling_data(data,"albumin")
# Age_Scaled=Scaling_data(data,"age")

# Diabetes Dataset
# pregnancies_scaled=Scaling_data(data2,"Pregnancies")
# glucose_scaled=Scaling_data(data2,"Glucose")
# bloodpressure_scaled=Scaling_data(data2,"BloodPressure")
# skinthickness_scaled=Scaling_data(data2,"SkinThickness")
# insulin_scaled=Scaling_data(data2,"Insulin")
# bmi_scaled=Scaling_data(data2,"BMI")
# DPF_scaled=Scaling_data(data2,"DiabetesPedigreeFunction")
# age_scaled=Scaling_data(data2,"Age")


# In[6]:


# writer = pd.ExcelWriter('C:/Users/pc/Desktop/hepatitis_edited.xlsx')
# data.to_excel(writer,'Sheet1')
# writer.save()


# In[7]:


# def mean(dataset,col_name):
#     mean=np.mean(dataset[col_name])
#     return mean

# def std(dataset,col_name):
#     std=np.std(dataset[col_name])
#     return std

# m=mean(data2,"Pregnancies")

# sd=std(data2,"Pregnancies")

# lowerlimit=m-sd;
# higherlimit=m+sd;
# print (lowerlimit)
# print(higherlimit)
# data2["Pregnancies"]


# for i in range(len(data2)):
#     if(data2["Pregnancies"][i] > lowerlimit)& (data2["Pregnancies"][i]< higherlimit):
        
#         print("yes")
#     else:
#         print("no")
    


# In[8]:


target_data=data["class"]
del data["class"]
data=data.values
target_data=target_data.values


# In[9]:


data[0]


# In[10]:


model=GaussianNB()
model.fit(data,target_data)
expected=target_data
predicted=model.predict(data)
confusion_matrix=metrics.confusion_matrix(expected,predicted)
accuracy=(confusion_matrix[0][0]+confusion_matrix[1][1])/(len(data))

print(accuracy*100)
print(confusion_matrix)


# In[11]:


result=model.predict([[50.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        1., 85., 18.,  4., 12.,  0.]])
result

