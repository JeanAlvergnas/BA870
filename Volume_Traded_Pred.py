import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Streamlit App Setup --------------------
st.set_page_config(page_title="Stock Volume Dashboard", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Top Traded Stocks", "User Input", "Prediction Output", "Feature Importance"])

# -------------------- Top Traded Stocks Tab --------------------
if page == "Top Traded Stocks":
    st.title("📊 Top 3 Most Traded Stocks (Past Month)")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    avg_volumes = []
    volume_data = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty and 'Volume' in df.columns:
            df['Date'] = df.index
            df['Ticker'] = ticker
            volume_data.append(df[['Date', 'Ticker', 'Volume']])
            avg_volumes.append((ticker, df['Volume'].mean()))

    avg_df = pd.DataFrame(avg_volumes, columns=['Ticker', 'AvgVolume'])
    top3 = avg_df.sort_values(by='AvgVolume', ascending=False).head(3)
    top3_tickers = top3['Ticker'].tolist()

    top3_data = [df for df in volume_data if df['Ticker'].iloc[0] in top3_tickers]

    if top3_data:
        combined_df = pd.concat(top3_data, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['Ticker'] = combined_df['Ticker'].astype(str)
        combined_df['Volume'] = pd.to_numeric(combined_df['Volume'].squeeze(), errors='coerce')
        combined_df.dropna(subset=['Volume'], inplace=True)

        if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns):
            pivot_df = combined_df.pivot_table(index='Date', columns='Ticker', values='Volume')
            st.line_chart(pivot_df)
        else:
            st.warning("Missing expected columns for plotting.")
    else:
        st.warning("No volume data found for the selected tickers.")

# -------------------- User Input Tab --------------------
elif page == "User Input":
    st.header("Select Stock and Time Range")
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

# -------------------- Prediction Output Tab --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")

    if 'user_data' in locals() and not user_data.empty:
        user_data = user_data.dropna(subset=["Volume"])

        random_factors = pd.Series(np.random.uniform(0.95, 1.05, size=user_data.shape[0]), index=user_data.index)
        predicted = user_data['Volume'].shift(1).fillna(method='bfill') * random_factors
        user_data['Predicted Volume'] = predicted

        st.subheader("Actual vs Predicted Volume")
        fig, ax = plt.subplots()
        ax.plot(user_data.index, user_data['Volume'], label='Actual Volume')
        ax.plot(user_data.index, user_data['Predicted Volume'], label='Predicted Volume', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Load stock data in the 'User Input' tab to view prediction.")

# -------------------- Feature Importance Tab --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Simulated)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)
    st.markdown("_Note: Replace simulated predictions and importances with real model outputs when ready._")

