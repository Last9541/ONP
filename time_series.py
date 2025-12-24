import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def windows(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


df = pd.read_csv("Gold Price.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

price_col = 'Price'
series = df[price_col]


train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

diff_series = train.diff().dropna()


sarimax = SARIMAX(train,order=(1, 1, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)

sarimax_fit = sarimax.fit(disp=False)

sarimax_pred = sarimax_fit.forecast(len(test))
sarimax_rmse = sqrt(mean_squared_error(test, sarimax_pred))

print(f"SARIMAX RMSE: {sarimax_rmse:.2f}")



scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.values.reshape(-1, 1))

window = 30
X, y = windows(scaled, window)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape((X_train.shape[0], window, 1))
X_test = X_test.reshape((X_test.shape[0], window, 1))

lstm = Sequential([LSTM(50, input_shape=(window, 1)),Dense(1)])

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

lstm_pred = lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_rmse = sqrt(mean_squared_error(y_test_inv, lstm_pred))
print(f"LSTM RMSE: {lstm_rmse:.2f}")


plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, label="Real price")

plt.plot(test.index,sarimax_pred,label="SARIMAX prediction")

plt.plot(series.index[-len(lstm_pred):],lstm_pred,label="LSTM prediction")

plt.legend()
plt.title("Gold Price Prediction")
plt.show()
