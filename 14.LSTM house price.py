'''
A financial analyst wants to model house price trends over increasing area numbers of 
rooms. Use an LSTM-based Recurrent Neural Network to predict the next house price 
value based on historical patterns in sorted data. 
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Generate Synthetic Data
# ------------------------------
np.random.seed(42)
area = np.linspace(500, 5000, 200)  # Increasing area
rooms = np.linspace(1, 10, 200) + np.random.normal(0, 0.5, 200)
price = 50000 + (area * 30) + (rooms * 10000) + np.random.normal(0, 20000, 200)

df = pd.DataFrame({
    'area': area,
    'rooms': rooms,
    'price': price
})

# Sort by area to simulate progression
df = df.sort_values(by='area').reset_index(drop=True)

# ------------------------------
# Step 2: Preprocessing & Sequence Creation
# ------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['area', 'rooms', 'price']])

sequence_length = 10
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length, :2])  # area and rooms as features
    y.append(scaled_data[i+sequence_length, 2])     # next price

X, y = np.array(X), np.array(y)

# ------------------------------
# Step 3: Define LSTM Model
# ------------------------------
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 2)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ------------------------------
# Step 4: Train Model
# ------------------------------
history = model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# ------------------------------
# Step 5: Predictions
# ------------------------------
predicted = model.predict(X)
true_price = scaler.inverse_transform(np.hstack((X[:, -1, :], y.reshape(-1,1))))[:, 2]
pred_price = scaler.inverse_transform(np.hstack((X[:, -1, :], predicted)))[:, 2]

# ------------------------------
# Step 6: Plot Results
# ------------------------------
plt.figure(figsize=(10, 5))
plt.plot(true_price, label='Actual Price')
plt.plot(pred_price, label='Predicted Price')
plt.legend()
plt.title("House Price Prediction using LSTM")
plt.xlabel("Sample")
plt.ylabel("Price")
plt.grid(True)
plt.show()
