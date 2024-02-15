#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[18]:


deposit= pd.read_csv('termdeposit_test.csv')


# In[19]:


deposit.head()


# In[20]:


deposit.isnull().sum()


# In[21]:


deposit.duplicated()


# In[14]:


# Perform one-hot encoding for categorical variables
deposit_encoded = pd.get_dummies(deposit)


# In[15]:


print(deposit_encoded.columns)


# In[24]:


# Separate features (X) and target variable (y)
X = deposit_encoded.drop('pdays', axis=1)
y = deposit_encoded['pdays']


# In[26]:


#split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


# initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[28]:


#train the model
model.fit(X_train, Y_train)


# In[29]:


# make predictions on the testing set 
predictions = model.predict(X_test)


# In[31]:


# Evalluate the model
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:




