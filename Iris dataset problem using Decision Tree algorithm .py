#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


dataset=pd.read_csv(r'C:\Data Scientist\Interview\iris.csv')


# In[68]:


dataset.head()


# In[69]:


dataset.info()


# In[70]:


dataset.shape


# In[71]:


dataset.isnull().sum()


# In[72]:


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False)


# In[73]:


sns.pairplot(dataset,hue='iris')


# In[74]:


X=dataset.iloc[:,[0,1,2,3]]
y=dataset.iloc[:,[4]]


# In[75]:


X.head()


# In[76]:


y.head()


# In[77]:


#Build the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test


# In[81]:


#The above code will create the empty model. Inorder to provide the operations to the model we should train them.
from sklearn import tree
classifier=tree.DecisionTreeClassifier()

#At this point,We have just made the model.But it cannot able to predict whether the given flower belongs to which species 
#of Iris .If our model has to predict the flower,We have to train the model with the Features and the Labels.


# In[82]:


#Train the Model.
#We can train the model with fit function.
classifier.fit(X_train,y_train)


# In[83]:


#4.Make predictions:
#Predictions can be done with predict function
predictions=classifier.predict(X_test)


# In[84]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[ ]:




