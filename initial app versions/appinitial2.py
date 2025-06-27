import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import datetime

st.header('📈 Stock Price Predictor Using RNN')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
end_date = st.date_input("Enter End Date", value=datetime.date.today())
future_type = st.selectbox("Predict into the future by:", ['Days', 'Months'])
future_count = st.number_input(f'Enter number of {future_type.lower()} to predict', min_value=1, value=30)

start_date = '2012-01-01'
data = yf.download(stock, start=start_date, end=str(end_date))

if data.empty:
    st.error("❌ No data found. Check the stock symbol or date range.")
    st.stop()

st.subheader('📊 Historical Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test_full)

def plot_ma_chart(title, *mas):
    fig = plt.figure(figsize=(10, 6))
    for ma_data, label, color in mas:
        plt.plot(ma_data, color, label=label)
    plt.plot(data.Close, 'g', label='Original')
    plt.title(title)
    plt.legend()
    st.pyplot(fig)

st.subheader('📈 Moving Averages')
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

plot_ma_chart("MA50", (ma_50, 'MA50', 'r'))
plot_ma_chart("MA50 vs MA100", (ma_50, 'MA50', 'b'), (ma_100, 'MA100', 'r'))
plot_ma_chart("MA100 vs MA200", (ma_100, 'MA100', 'b'), (ma_200, 'MA200', 'r'))

x_test, y_test = [], []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

model_path = r'C:\Users\taish\OneDrive\Documents\mypythonprojects\stock prize detection\stock pred model.keras'
model = load_model(model_path)

predicted = model.predict(x_test)
scale = 1 / scaler.scale_[0]
predicted = predicted * scale
y_test = y_test * scale

st.subheader('✅ Actual Price vs Predicted Price')
test_dates = data_test.index[-len(predicted):]

predicted = predicted.flatten()
y_test = y_test.flatten()

fig4 = plt.figure(figsize=(10, 6))
plt.plot(test_dates, predicted, 'r', label='Predicted Price')
plt.plot(test_dates, y_test, 'g', label='Actual Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)

st.subheader('🔮 Future Price Prediction')

last_100_days = data_test_scale[-100:]
future_input = last_100_days.reshape(1, -1)
temp_input = list(future_input[0])
future_predictions = []

n_steps = 100
num_days = future_count if future_type == 'Days' else future_count * 30

for i in range(num_days):
    if len(temp_input) > 100:
        temp_input = temp_input[1:]
    input_seq = np.array(temp_input[-100:]).reshape(1, n_steps, 1)
    next_price = model.predict(input_seq, verbose=0)[0][0]
    temp_input.append(next_price)
    future_predictions.append(next_price)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = future_predictions * scale

last_date = pd.to_datetime(data.index[-1])
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days)

fig_future = plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, label='Future Predicted Price', color='orange')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.title(f' Forecast for next {future_count} {future_type.lower()}')
plt.legend()
st.pyplot(fig_future)
