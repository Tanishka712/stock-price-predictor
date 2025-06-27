# 📈 Stock Price Predictor Using RNN

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-price-predict.streamlit.app/)

A web-based stock price prediction app built with **Streamlit**, **Keras**, and **Plotly**. This project leverages historical stock data and an RNN (LSTM-based) model to forecast future stock prices. The app supports both short- and long-term forecasts and presents visual insights using interactive plots.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://stock-price-predict.streamlit.app/)

> Or run it locally using the command:
```bash
streamlit run app.py
```

---

## 🧠 Project Highlights

* RNN Model with LSTM Layers trained on historical stock data (2012–2022)
* Interactive UI with Streamlit for real-time prediction
* Auto-fetching of stock data using yFinance
* Moving Averages Visualizations (MA50, MA100, MA200)
* Prediction for future days or months
* Performance Metrics (MSE, RMSE, MAE, R² Score)

---

## 📂 Project Structure

```
.
├── initial app versions/         # Old app versions (if any)
├── model/                        # Contains training notebook and saved model
│   └── stock_pred_model.keras    # Trained model
├── app.py                        # Streamlit web application
├── README.md                     # Project documentation
├── requirements.txt              # Project requirements
```

---

## 🧾 How to Use Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Tanishka712/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install Dependencies

Make sure you have Python 3.7+ and install required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 💡 Features

### 🔍 Input Section

* Enter stock symbol (e.g., AAPL, GOOG)
* Select end date for analysis
* Choose prediction type: **Days** or **Months**
* Enter number of future intervals to predict

### 📊 Output Section

* Displays historical stock data
* Shows moving averages (MA50, MA100, MA200)
* Actual vs Predicted Price interactive graph
* Future predictions with Plotly timeline
* Retry mechanism in case stock data fetch fails

---

## 🧠 Model Architecture

```
LSTM (50 units, relu) + Dropout(0.2)
→ LSTM (60 units) + Dropout(0.3)
→ LSTM (80 units) + Dropout(0.4)
→ LSTM (120 units) + Dropout(0.5)
→ Dense(1)
```

* Trained for 50 epochs using `adam` optimizer and `mean_squared_error` loss  
* Scaled using `MinMaxScaler`  
* Window size: 100 time steps

---

## 📊 Evaluation Metrics

The model was evaluated using:

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **R² Score**

These help assess how closely predictions match actual stock values.

---

## 📈 Example Output Graphs

* Actual vs Predicted Price
* Forecast for Next N Days or Months
* Moving Averages (MA50, MA100, MA200)

<!-- Add screenshots here if available -->

---

## 📌 Notes

* The model works best on large-cap, actively traded stocks (e.g., GOOG, AAPL)
* Does **not guarantee** investment decisions—this is a **data science project**, not financial advice
* Future predictions use extrapolation based on past 100 days

---

## 🙌 Credits

* [Streamlit](https://streamlit.io/)
* [Keras](https://keras.io/)
* [yFinance](https://pypi.org/project/yfinance/)
* [Plotly](https://plotly.com/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/)

---

## 📬 Contact

Made with ❤️ by **Tanishka Nagawade**  
Feel free to connect or contribute!

