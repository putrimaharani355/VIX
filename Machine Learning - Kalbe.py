#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

date_format = '%d/%m/%Y'

# Step 1: Load and preprocess the data
# Misalkan Anda memiliki empat file CSV dengan data sesuai atribut yang telah dijelaskan sebelumnya
# Load data dari file CSV ke dalam DataFrame dan sesuaikan tipe data
customer_data = pd.read_csv('Case Study - Customer.csv', delimiter=';')
store_data = pd.read_csv('Case Study - Store.csv', delimiter=';')
product_data = pd.read_csv('Case Study - Product.csv', delimiter=';')
transaction_data = pd.read_csv('Case Study - Transaction.csv', delimiter=';')

# display(transaction_data)
# Melakukan data cleansing (jika diperlukan) dan penyesuaian tipe data
# Misalnya, mengubah kolom 'Date' menjadi tipe data datetime
transaction_data['Date'] = pd.to_datetime(transaction_data['Date'], format=date_format)

# Step 2: Merge data
# Menggabungkan data dari berbagai tabel berdasarkan atribut yang sesuai
merged_data = pd.merge(transaction_data, product_data, on='ProductID')
merged_data = pd.merge(merged_data, store_data, on='StoreID')
merged_data = pd.merge(merged_data, customer_data, on='CustomerID')

# Step 3: Membuat data baru untuk regression dengan groupby date dan qty di sum
daily_total_quantity = merged_data.groupby('Date')['Qty'].sum().reset_index()

# Step 4: Time series analysis dengan ARIMA
# Misalnya, Anda akan menggunakan ARIMA dengan order (1, 1, 1)
# Anda dapat menyesuaikan order berdasarkan karakteristik data Anda
# Train the ARIMA model using the daily total quantity data
train_size = int(len(daily_total_quantity) * 0.8)
train, test = daily_total_quantity[:train_size], daily_total_quantity[train_size:]

# Fit the ARIMA model
order = (1, 1, 1)
model = ARIMA(train['Qty'], order=order)
fitted_model = model.fit()

# Step 5: Make predictions dan evaluate model
# Menggunakan fitted ARIMA model untuk membuat prediksi pada test set
predictions = fitted_model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Evaluasi performa model dengan menggunakan metrik, misalnya Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions - test['Qty']))

# Step 6: Visualisasi hasil prediksi (opsional)
# Plot actual vs. predicted values untuk mengevaluasi performa model secara visual
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

# Step 1: Load and preprocess the data
# Misalkan Anda memiliki empat file CSV dengan data sesuai atribut yang telah dijelaskan sebelumnya
# Load data dari file CSV ke dalam DataFrame dan sesuaikan tipe data

customer_data = pd.read_csv('Case Study - Customer.csv', delimiter=';')
store_data = pd.read_csv('Case Study - Store.csv', delimiter=';')
product_data = pd.read_csv('Case Study - Product.csv', delimiter=';')
transaction_data = pd.read_csv('Case Study - Transaction.csv', delimiter=';')

# Melakukan data cleansing (jika diperlukan) dan penyesuaian tipe data

# Assuming 'date_column' is the column containing dates in the DataFrame
date_format = '%d/%m/%Y'
transaction_data['Date'] = pd.to_datetime(transaction_data['Date'], format=date_format)

# Step 2: Merge data
# Menggabungkan data dari berbagai tabel berdasarkan atribut yang sesuai
merged_data = pd.merge(transaction_data, product_data, on='ProductID')
merged_data = pd.merge(merged_data, store_data, on='StoreID')
merged_data = pd.merge(merged_data, customer_data, on='CustomerID')


# Step 3: Membuat data baru untuk clustering dengan groupby customerID dan aggregasi Transaction ID, Qty, Total Amount
customer_clusters = merged_data.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()

# Step 4: Feature Scaling
# Sebelum melakukan clustering, lakukan feature scaling untuk membuat skala data menjadi sejajar
scaler = StandardScaler()
customer_clusters_scaled = scaler.fit_transform(customer_clusters[['TransactionID', 'Qty', 'TotalAmount']])

# Step 5: Clustering dengan KMeans
# Misalnya, Anda akan menggunakan 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
customer_clusters['Cluster'] = kmeans.fit_predict(customer_clusters_scaled)

# Step 6: Visualisasi hasil clustering
# Misalnya, Anda akan plot scatter plot dari dua fitur terpilih dan mewarnai sesuai dengan kluster

plt.figure(figsize=(10, 6))
plt.scatter(customer_clusters['Qty'], customer_clusters['TotalAmount'], c=customer_clusters['Cluster'], cmap='rainbow')
plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.title('KMeans Clustering - Quantity vs Total Amount')
plt.show()


# In[ ]:




