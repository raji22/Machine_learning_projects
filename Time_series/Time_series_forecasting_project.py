import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

#Load dataset
df = pd.read_csv("train.csv")
print(df.head())
print(df.info())

#Data Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'],dayfirst=True)
df = df.sort_values('Order Date')
print(df.head())

#Checking duplicates
print(df.duplicated().sum())

#Convert Dates & Summarize Sales
df['Order Date'] = pd.to_datetime(df['Order Date'],dayfirst=True)
sales_daily = df.groupby('Order Date')['Sales'].sum().reset_index()
print(sales_daily)

df['Month'] = df['Order Date'].dt.to_period('M')
sales_monthly = df.groupby('Month')['Sales'].sum().reset_index()
print(sales_monthly)

#Visualize Sales Trend
sales_monthly.plot(x='Month', y='Sales', figsize=(10,5))
plt.title("Monthly Sales Trend")
plt.show()

#Monthly ARIMA Forecast
train = sales_monthly.iloc[:-6]
test = sales_monthly.iloc[-6:]

model = ARIMA(train['Sales'], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

#Evaluate Model
mae = mean_absolute_error(test['Sales'], forecast)
print("MAE:", mae)