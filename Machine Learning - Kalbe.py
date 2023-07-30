#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

date_format = '%d/%m/%Y'

customer_data = pd.read_csv('Case Study - Customer.csv', delimiter=';')
store_data = pd.read_csv('Case Study - Store.csv', delimiter=';')
product_data = pd.read_csv('Case Study - Product.csv', delimiter=';')
transaction_data = pd.read_csv('Case Study - Transaction.csv', delimiter=';')

transaction_data['Date'] = pd.to_datetime(transaction_data['Date'], format=date_format)

merged_data = pd.merge(transaction_data, product_data, on='ProductID')
merged_data = pd.merge(merged_data, store_data, on='StoreID')
merged_data = pd.merge(merged_data, customer_data, on='CustomerID')

daily_total_quantity = merged_data.groupby('Date')['Qty'].sum().reset_index()

train_size = int(len(daily_total_quantity) * 0.8)
train, test = daily_total_quantity[:train_size], daily_total_quantity[train_size:]

order = (1, 1, 1)
model = ARIMA(train['Qty'], order=order)
fitted_model = model.fit()

predictions = fitted_model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

mae = np.mean(np.abs(predictions - test['Qty']))

plt.figure(figsize=(10, 6))
plt.plot(test['Date'], test['Qty'], label='Actual')
plt.plot(test['Date'], predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.legend()
plt.title('ARIMA Model - Actual vs. Predicted')
plt.show()


# In[28]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

customer_data = pd.read_csv('Case Study - Customer.csv', delimiter=';')
store_data = pd.read_csv('Case Study - Store.csv', delimiter=';')
product_data = pd.read_csv('Case Study - Product.csv', delimiter=';')
transaction_data = pd.read_csv('Case Study - Transaction.csv', delimiter=';')

date_format = '%d/%m/%Y'
transaction_data['Date'] = pd.to_datetime(transaction_data['Date'], format=date_format)

merged_data = pd.merge(transaction_data, product_data, on='ProductID')
merged_data = pd.merge(merged_data, store_data, on='StoreID')
merged_data = pd.merge(merged_data, customer_data, on='CustomerID')


customer_clusters = merged_data.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()

scaler = StandardScaler()
customer_clusters_scaled = scaler.fit_transform(customer_clusters[['TransactionID', 'Qty', 'TotalAmount']])

kmeans = KMeans(n_clusters=5, random_state=42)
customer_clusters['Cluster'] = kmeans.fit_predict(customer_clusters_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(customer_clusters['Qty'], customer_clusters['TotalAmount'], c=customer_clusters['Cluster'], cmap='rainbow')
plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.title('KMeans Clustering - Quantity vs Total Amount')
plt.show()


# In[ ]:




