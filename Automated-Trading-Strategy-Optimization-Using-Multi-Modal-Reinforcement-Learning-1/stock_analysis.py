import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os 
CSV_FILE_PATH = 'nse_data_2025.csv' 


data = pd.read_csv(CSV_FILE_PATH)
print(f"Successfully loaded data from {CSV_FILE_PATH}")


if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

ticker_to_process = 'RELIANCE.NS' 
if 'Stock' in data.columns and not data['Stock'].nunique() == 1:
    data = data[data['Stock'] == ticker_to_process].copy()
    print(f"Filtered data for stock: {ticker_to_process}")
elif 'Stock' in data.columns and data['Stock'].nunique() == 1:
    print(f"Processing data for single stock: {data['Stock'].iloc[0]}")
    data.drop(columns=['Stock'], inplace=True)

data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore', inplace=True)

print("\n--- Data Head ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe())

data.dropna(inplace=True)
if data.empty:
    print("Error: DataFrame is empty after dropping NaNs. Cannot proceed with analysis.")
    exit()

features = ['Open', 'High', 'Low', 'Close', 'Volume']

print("\n--- Part 2: Linear Regression ---")

data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)

if data.empty:
    print("Error: DataFrame is empty after creating Next_Close and dropping NaNs. Cannot proceed with Linear Regression.")
    exit()

X_lr = data[features]
y_lr = data["Next_Close"]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)

mse = mean_squared_error(y_test_lr, y_pred_lr)
print(f"Linear Regression MSE: {mse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_lr.index, y_test_lr.values, label='Actual Close Price')
plt.plot(y_test_lr.index, y_pred_lr, label='Predicted Close Price')
plt.title('Linear Regression - Next Day Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Part 3: Logistic Regression ---")

data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)

X_log = data[features]
y_log = data["Target"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, shuffle=False)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_log, y_train_log)
y_pred_log = log_model.predict(X_test_log)

accuracy_log = accuracy_score(y_test_log, y_pred_log)
conf_matrix_log = confusion_matrix(y_test_log, y_pred_log)

print(f"Logistic Regression Accuracy: {accuracy_log:.4f}")
print("Logistic Regression Confusion Matrix:\n", conf_matrix_log)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0 (Down/Same)', 'Predicted 1 (Up)'],
            yticklabels=['Actual 0 (Down/Same)', 'Actual 1 (Up)'])
plt.title('Logistic Regression Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n--- Part 4: K-Nearest Neighbors (KNN) Classification ---")

X_knn = data[features]
y_knn = data["Target"]

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, shuffle=False)

print("Evaluating KNN for various K values:")
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn.predict(X_test_knn)
    acc_knn = accuracy_score(y_test_knn, y_pred_knn)
    print(f"K={k}, Accuracy={acc_knn:.4f}")
