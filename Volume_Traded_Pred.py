# stock_volume_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# -------------------- Config --------------------
st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigate")
pages = ["Top Traded Stocks", "User Input", "Prediction Output", "Feature Importance"]
page = st.sidebar.radio("Go to", pages)

# -------------------- Page 1: Top 3 Most Traded Stocks (Last Month) --------------------
if page == "Top Traded Stocks":
    st.title("📊 Stock Volume Prediction App")
    st.header("Top 3 Most Traded Stocks Over the Past Month")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=30)

    avg_volumes = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            if not df.empty:
                avg_volume = df['Volume'].mean()
                avg_volumes.append((ticker, avg_volume))
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")

    avg_df = pd.DataFrame(avg_volumes, columns=["Ticker", "AvgVolume"])
    avg_df = avg_df.dropna(subset=["AvgVolume"])
    avg_df["AvgVolume"] = pd.to_numeric(avg_df["AvgVolume"], errors="coerce")
    avg_df = avg_df.sort_values("AvgVolume", ascending=False)
    top3_tickers = avg_df["Ticker"].head(3).tolist()

    volume_data = []
    for ticker in top3_tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            df = df[['Volume']].copy()
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df = df[df['Volume'].notna()]
            df['Ticker'] = ticker
            df['Date'] = df.index
            volume_data.append(df)

    if volume_data:
        combined_df = pd.concat(volume_data, ignore_index=True)
        st.write("Before pivot:", combined_df.head())
        st.write("Shape:", combined_df.shape)

        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['Ticker'] = combined_df['Ticker'].astype(str)
        combined_df['Volume'] = pd.to_numeric(combined_df['Volume'], errors='coerce')

        if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns) and combined_df['Volume'].ndim == 1:
            try:
                pivoted = combined_df.pivot(index='Date', columns='Ticker', values='Volume')
                st.line_chart(pivoted)
            except Exception as e:
                st.error(f"Pivot failed: {e}")
        else:
            st.error("Data not in correct shape or Volume is not 1D.")
    else:
        st.warning("No volume data available to display.")

# -------------------- Page 2: User Input --------------------
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

# -------------------- Page 3: Prediction Output --------------------
elif page == "Prediction Output":
    st.header("Prediction Model Output")

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

# -------------------- Page 4: Feature Importance --------------------
elif page == "Feature Importance":
    st.header("Feature Importance (Model Explainability)")
    features = ['Lag_1_Volume', 'Price_Change', 'Moving_Avg_7d', 'RSI', 'MACD']
    importances_raw = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances_raw})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title("Feature Importance (Simulated)")
    st.pyplot(fig)
    st.markdown("_Note: Replace simulated predictions and importances with real model outputs when ready._")
