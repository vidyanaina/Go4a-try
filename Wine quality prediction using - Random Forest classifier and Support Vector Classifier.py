#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#In this notebook, First I have done some exploration on the data using matplotlib and seaborn. Then, I used different 
#classifier models to predict the quality of the wine. Models used are:

#1. Random Forest Classifier

#2. Stochastic Gradient Descent Classifier and

#3. Support Vector Classifier(SVC)

#Then I used cross validation evaluation technique to optimize the model performance.

#1. Grid Search CV and

#2. Cross Validation Score


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset=pd.read_csv(r'C:\Data Scientist\Interview\winequality-red.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


dataset.isnull().sum()


# In[8]:


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False)


# In[ ]:


#Let's do some plotting to know how the data columns are distributed in the dataset


# In[9]:


#Here we see that fixed acidity does not give any specification to classify the quality.

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = dataset)


# In[10]:


#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = dataset)


# In[11]:


#Composition of citric acid go higher as we go higher in the quality of the wine

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = dataset)


# In[12]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = dataset)


# In[13]:


#Composition of chloride also go down as we go higher in the quality of the wine

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = dataset)


# In[14]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = dataset)


# In[15]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = dataset)


# In[16]:


#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = dataset)


# In[17]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = dataset)


# In[ ]:


#Preprocessing Data for performing Machine learning algorithms


# In[18]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality


# In[19]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)


# In[20]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[21]:


#Bad becomes 0 and good becomes 1 
dataset['quality'] = label_quality.fit_transform(dataset['quality'])


# In[22]:


dataset['quality'].value_counts()


# In[23]:


sns.countplot(dataset['quality'])


# In[24]:


#Now seperate the dataset as response variable and feature variabes
X = dataset.drop('quality', axis = 1)
y = dataset['quality']


# In[25]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[26]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[27]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


Our training and testing data is ready now to perform machine learning algorithm


# In[ ]:


Random Forest Classifier


# In[28]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[29]:


#Let's see how our model performed
print(classification_report(y_test, pred_rfc))


# In[ ]:


#Random forest gives the accuracy of 87%


# In[30]:


#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))


# In[31]:


#Stochastic Gradient Decent Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# In[32]:


print(classification_report(y_test, pred_sgd))


# In[ ]:


84% accuracy using stochastic gradient descent classifier


# In[33]:


print(confusion_matrix(y_test, pred_sgd))


# In[34]:


#Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[35]:


print(classification_report(y_test, pred_svc))


# In[ ]:


#Support vector classifier gets 86%


# In[36]:


#Let's try to increase our accuracy of models


# In[37]:


#Grid Search CV


# In[38]:


#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)


# In[39]:


grid_svc.fit(X_train, y_train)


# In[40]:


#Best parameters for our svc model
grid_svc.best_params_


# In[41]:


#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))


# In[ ]:


#SVC improves from 86% to 90% using Grid Search CV


# In[ ]:


#Cross Validation Score for random forest and SGD


# In[42]:


#Now lets try to do some evaluation for random forest model using cross validation.
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()


# In[ ]:


#Random forest accuracy increases from 87% to 91 % using cross validation score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




