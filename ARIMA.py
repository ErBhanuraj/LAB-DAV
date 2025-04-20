import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Month_Value_1.csv")
df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')
df.set_index('Period', inplace=True)
df = df.sort_index()
df['Revenue'] = df['Revenue'].interpolate(method='time')

plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['Revenue'])
plt.title('Revenue Over Time')
plt.grid(True)
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    print("Stationary" if result[1] <= 0.05 else "Not Stationary")

adf_test(df['Revenue'])
df['Revenue_diff'] = df['Revenue'].diff().dropna()
adf_test(df['Revenue_diff'])

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['Revenue_diff'], ax=ax[0])
plot_pacf(df['Revenue_diff'], ax=ax[1])
plt.show()

model = ARIMA(df['Revenue'], order=(2, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Revenue': forecast.values})

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Revenue'], label="Actual", color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Revenue'], label="Forecast", color='red', linestyle='dashed')
plt.title("ARIMA Forecast for Revenue")
plt.legend()
plt.grid(True)
plt.show()
