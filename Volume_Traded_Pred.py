# stock_volume_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import time

# -------------------- Config --------------------
st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# -------------------- Window 1: Top 5 Most Traded Stocks (1-Month) --------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time

st.title("📊 Stock Volume Prediction App")
st.header("1. Top 5 Most Traded Stocks Over the Past Month")

API_KEY = "CXW22KLIBXMMW6KU"  # Replace this with your actual API key
ALPHA_URL = "https://www.alphavantage.co/query"

# 🔽 Shorter list of tickers for faster testing
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']

# Function to get average daily volume using Alpha Vantage
def get_avg_volume(ticker):
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": API_KEY
    }
    response = requests.get(ALPHA_URL, params=params)
    data = response.json()
    try:
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        # Filter to last 30 days only
        last_30 = df[df.index >= pd.to_datetime("today") - pd.Timedelta(days=30)]
        return last_30['6. volume'].mean()
    except Exception as e:
        st.warning(f"Error processing {ticker}: {e}")
        return None

# Calculate average volume for each ticker
avg_volumes = []
for i, tkr in enumerate(tickers):
    vol = get_avg_volume(tkr)
    if vol is not None:
        avg_volumes.append((tkr, vol))
    time.sleep(12)  # Respect Alpha Vantage free tier limit

# Pick top 5 tickers based on recent 1-month average volume
top5_tickers = [t[0] for t in sorted(avg_volumes, key=lambda x: x[1], reverse=True)[:5]]

# Use yfinance to get weekly volume for charting (optional: switch to daily)
volume_data = []
end_date = pd.to_datetime("today")
start_date = end_date - pd.Timedelta(days=30)

for ticker in top5_tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
    if not df.empty and 'Volume' in df.columns:
        df = df[['Volume']].copy()
        df['Ticker'] = ticker
        df['Date'] = df.index
        volume_data.append(df)

# Combine and plot
if volume_data:
    combined_df = pd.concat(volume_data, ignore_index=True)
    combined_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')

    st.subheader("Top 5 Stocks by Average Daily Volume (Last Month)")
    st.line_chart(combined_df)
else:
    st.warning("No volume data available to display.")

# -------------------- Window 2: User Input --------------------
st.header("2. Select Stock and Time Range")
ticker = st.text_input("Enter stock ticker:", value="TSLA")
start = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
end = st.date_input("End date", value=pd.to_datetime("today"))
interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'])

if st.button("Load Stock Data"):
    user_data = yf.download(ticker, start=start, end=end, interval=interval)
    if not user_data.empty:
        st.success("Data loaded successfully!")
        st.dataframe(user_data.tail())

        st.subheader("📈 Volume Chart")
        st.line_chart(user_data['Volume'])
    else:
        st.error("No data found for the selected inputs.")

# -------------------- Window 3: Prediction Output --------------------
st.header("3. Prediction Model Output")

if 'user_data' in locals() and not user_data.empty:
    user_data = user_data.dropna(subset=["Volume"])

    random_factors_raw = np.random.uniform(low=0.95, high=1.05, size=user_data.shape[0])
    random_factors = pd.Series(random_factors_raw, index=user_data.index)

    predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
    user_data['Predicted Volume'] = predicted

    st.subheader("Actual vs Predicted Volume")
    fig, ax = plt.subplots()
    ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
    ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Load data to see prediction output.")

# -------------------- Window 4: Feature Importance --------------------
st.header("4. Feature Importance (Model Explainability)")

features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
importances_raw = np.random.dirichlet(np.ones(len(features)), size=1)[0]
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances_raw})
importance_df.sort_values(by='Importance', ascending=True, inplace=True)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
ax.set_title("Feature Importance (Simulated)")
st.pyplot(fig)

st.markdown("_Note: Replace simulated predictions and importances with real model outputs when ready._")        