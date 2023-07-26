#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

np.random.seed(42)

# Generating synthetic data for churn model
num_samples = 100
age = np.random.randint(18, 65, num_samples)
monthly_spend = np.random.uniform(100, 1000, num_samples)
num_transactions = np.random.randint(1, 20, num_samples)
churn_prob = 0.3 + (age / 100) + (monthly_spend / 5000) - (num_transactions / 100)
churn = np.random.rand(num_samples) < churn_prob

data = pd.DataFrame({
    'Age': age,
    'MonthlySpend': monthly_spend,
    'NumTransactions': num_transactions,
    'Churn': churn.astype(int)
})


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Splitting the data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the churn prediction model (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting churn on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




