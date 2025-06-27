import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import datetime
import matplotlib.pyplot as plt

import time

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.header(" Stock Price Predictor Using RNN")


stock = st.text_input("Enter Stock Symbol (e.g., GOOG, TSLA, AAPL)", value="GOOG")
end_date = st.date_input("Select End Date", value=datetime.date.today())
future_type = st.selectbox("Predict into the future by:", ["Days", "Months"])
future_count = st.number_input(f"Enter number of {future_type.lower()} to predict", min_value=1, value=30)

start_date = "2012-01-01"
max_retries = 3
data = None

for attempt in range(max_retries):
    try:
        data = yf.download(stock, start=start_date, end=str(end_date), timeout=30)
        if not data.empty:
            break
    except Exception as e:
        st.warning(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(5)

# --- Final Check ---
if data is None or data.empty:
    st.error(" Failed to fetch stock data after multiple attempts. Please try again later or check the symbol/date.")

st.subheader(' Historical Stock Data')
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

st.subheader(' Moving Averages')
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

model_path = 'model/stock pred model.keras'
model = load_model(model_path)


predicted = model.predict(x_test)
scale = 1 / scaler.scale_[0]
predicted = predicted * scale
y_test = y_test * scale

test_dates = data.index[-len(predicted):]
predicted = predicted.flatten()
y_test = y_test.flatten()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=test_dates, y=predicted, name='Predicted Price', line=dict(color='orange')))
fig4.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual Price', line=dict(color='green')))

fig4.update_layout(
    legend=dict(font=dict(color='black'))
)


fig4.update_layout(
    title=dict(text='Actual Price vs Predicted Price', font=dict(color='black')),
    xaxis_title=dict(text='Date', font=dict(color='black')),
    yaxis_title=dict(text='Price', font=dict(color='black')),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickfont=dict(color='black'),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label='All')
            ]),
            font=dict(color='white')  
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )
)

st.plotly_chart(fig4, use_container_width=True)


from datetime import timedelta

st.subheader('🔮 Future Price Prediction')

last_100_days = data_test_scale[-100:]
future_input = last_100_days.reshape(1, -1)
temp_input = list(future_input[0])
future_predictions = []

n_steps = 100
num_days = future_count if future_type == 'Days' else future_count * 30

for i in range(num_days):
    if len(temp_input) > n_steps:
        temp_input = temp_input[1:]
    input_seq = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
    next_price_scaled = model.predict(input_seq, verbose=0)[0][0]
    temp_input.append(next_price_scaled)
    future_predictions.append(next_price_scaled)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)  # Correct inverse scaling

last_date = pd.to_datetime(data.index[-1])
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days)


fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), name='Future Price', line=dict(color='purple')))

fig_future.update_layout(
    title=dict(
        text=f'Forecast for next {future_count} {future_type.lower()}',
        font=dict(color='black')
    ),
    xaxis_title=dict(text='Date', font=dict(color='black')),
    yaxis_title=dict(text='Predicted Price', font=dict(color='black')),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickfont=dict(color='black'),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label='All')
            ]),
            font=dict(color='White')
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )
)


st.plotly_chart(fig_future, use_container_width=True)
