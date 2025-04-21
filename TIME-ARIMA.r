df<-read.csv("Month_Value_1.csv")
df
df<-na.omit(df)
df

ts_data<-ts(df$Revenue,start=c(2015,1),frequency=12)
ts_data
png("tsdata.png",width=800,height=600)
plot(ts_data)
dev.off()

de<-decompose(ts_data)
png("myplot2.png",width=800,height=600)
plot(de)
dev.off()

library(tseries)
adf<-adf.test(ts_data)
adf

d_data<-diff(ts_data)
d_data
png("diffplot.png",width=800,height=600)
plot(d_data)
dev.off()

de1<-decompose(d_data)
png("myplot3.png",width=800,height=600)
plot(de1)
dev.off()

png("acf_plot.png", width=800, height=600)
acf(d_data,lag.max=6)
dev.off()

png("pacf_plot.png", width=800, height=600)
pacf(d_data,lag.max=6)
dev.off()

library(forecast)
library(tseries)

# Fit ARIMA model
arima_model <- auto.arima(ts_data)

# Display model summary
summary(arima_model)

# Forecast for the next 12 months
forecast_values <- forecast(arima_model, h=12)

# Plot the forecast
png("forecast_plot.png", width=800, height=600)
plot(forecast_values)
dev.off()

# Print forecast values
print(forecast_values)

