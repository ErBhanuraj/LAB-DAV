import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("Month_Value_1.csv")
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
df.dropna(inplace=True)
df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Value'}, inplace=True)
df.set_index('Date', inplace=True)

# Plot original data
plt.figure(figsize=(15, 7))
plt.plot(df['Value'], color='blue')
plt.title("Time Series Plot")
plt.xlabel("Period")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# ADF test function
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    print("Stationary" if result[1] <= 0.05 else "Not Stationary")

adf_test(df['Value'])

# Differencing to make series stationary
df['Value_Diff'] = df['Value'].diff().dropna()
adf_test(df['Value_Diff'])

# Seasonal decomposition
decomposition = seasonal_decompose(df['Value'], model='additive', period=12)
decomposition.plot(figsize=(12, 8))
plt.show()

# ARIMA model fitting
model = ARIMA(df['Value'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting next 12 months
forecast = model_fit.forecast(steps=12)
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label="Original", color='blue')
plt.plot(pd.date_range(df.index[-1], periods=13, freq='M')[1:], forecast, label="Forecast", color='red')
plt.title("Time Series Forecasting using ARIMA")
plt.legend()
plt.grid(True)
plt.show()
