#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[16]:


job = pd.read_csv('termdeposit_train.csv')


# In[17]:


job.head()


# In[18]:


job.isnull().sum()


# In[26]:


job.duplicated()


# In[28]:


# Separate features (X) and target variable (y)
X = job.drop(['ID', 'subscribed'], axis=1)  # Features excluding 'ID' and 'subscribed'
y = job['subscribed']


# In[29]:


# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)


# In[30]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[31]:


# Initialize the random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[32]:


# Train the model
model.fit(X_train, y_train)


# In[33]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[34]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

