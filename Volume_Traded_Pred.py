# stock_volume_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------- Config --------------------
st.set_page_config(page_title="Stock Volume Prediction App", layout="wide")

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Top Traded Stocks",
    "User Input",
    "Prediction Output",
    "Feature Importance"
])

# -------------------- Tab 1: Top Traded Stocks --------------------
if page == "Top Traded Stocks":
    st.title("📊 Stock Volume Prediction App")
    st.header("Top 3 Most Traded Stocks Over the Past Month")

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']
    start_date = pd.to_datetime("today") - pd.Timedelta(days=30)
    end_date = pd.to_datetime("today")

    avg_volumes = []
    volume_data = []

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if not df.empty and "Volume" in df.columns:
            avg_volume = float(df["Volume"].mean())  # Ensure it's a float
            avg_volumes.append((ticker, avg_volume))

            df = df[["Volume"]].copy()
            df["Date"] = df.index
            df["Ticker"] = ticker
            volume_data.append(df)

    avg_df = pd.DataFrame(avg_volumes, columns=['Ticker', 'AvgVolume'])
    top3 = avg_df.sort_values(by='AvgVolume', ascending=False).head(3)

    top3_data = [df for df in volume_data if df['Ticker'].iloc[0] in top3['Ticker'].values]

    if top3_data:
        combined_df = pd.concat(top3_data, ignore_index=True)
        combined_df = combined_df[['Date', 'Ticker', 'Volume']]

        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['Ticker'] = combined_df['Ticker'].astype(str)
        combined_df['Volume'] = pd.to_numeric(combined_df['Volume'], errors='coerce')
        combined_df.dropna(subset=['Volume'], inplace=True)

        if {'Date', 'Ticker', 'Volume'}.issubset(combined_df.columns):
            pivot_df = combined_df.pivot_table(index='Date', columns='Ticker', values='Volume')
            st.line_chart(pivot_df)
        else:
            st.warning("Missing necessary columns for visualization.")
    else:
        st.warning("No data available for the top 3 stocks.")

# -------------------- Tab 2: User Input --------------------
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

# -------------------- Tab 3: Prediction Output --------------------
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

        st.metric(
            label="📊 Predicted Volume for Next Period",
            value=f"{user_data['Predicted Volume'].iloc[-1]:,.0f} shares"
        )

        st.caption("Note: This is a placeholder prediction.")
    else:
        st.info("Load data to see prediction output.")

# -------------------- Tab 4: Feature Importance --------------------
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
