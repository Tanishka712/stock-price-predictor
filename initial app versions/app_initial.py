# %%
import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt
# %%
model= load_model(r'C:\Users\taish\OneDrive\Documents\mypythonprojects\stock prize detection\stock pred model.keras')
# %%
import streamlit as st
# %%
st.header('Stock Price Predictor Using RNN')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end= '2023-01-29'
data = yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale= scaler.fit_transform(data_test)

st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1= plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Original')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('MA50 VS MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2= plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'b', label='MA50')
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(data.Close, 'g', label='Original')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('MA 100 VS MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3= plt.figure(figsize=(10,8))
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(ma_200_days, 'r', label='MA200')
plt.plot(data.Close, 'g', label='Original')
plt.legend()
plt.show()
st.pyplot(fig3)

x=[]
y=[]

for i in range (100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)
# %%

predict = model.predict(x)

scale= 1/scaler.scale_

predict = predict*scale

y= y*scale

st.subheader('Original Price VS Predicted Price')

fig4= plt.figure(figsize=(10,8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

st.subheader('📉 Future Price Prediction (Next 30 Days)')

future_input = data_test_scale[-100:].tolist()
future_pred = []
future_days = 30

for _ in range(future_days):
    current_input = np.array(future_input[-100:]).reshape(1, 100, 1)
    next_price = model.predict(current_input, verbose=0)[0][0]
    future_pred.append(next_price)
    future_input.append([next_price])

future_pred = np.array(future_pred).reshape(-1, 1)
future_pred = future_pred * scale

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

fig5 = plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_pred, label='Future Predicted Price', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig5)