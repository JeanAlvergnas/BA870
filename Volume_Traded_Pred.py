import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------- Window 1: Top 3 Most Traded Stocks Over the Past Month --------------------
st.title("📊 Stock Volume Prediction App")
st.header("Top 3 Most Traded Stocks Over the Past Month")

# Short list of tickers for demo purposes
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'NFLX', 'AMD']

# Fetch data and calculate average volume
avg_volumes = []
end_date = pd.to_datetime("today")
start_date = end_date - pd.Timedelta(days=30)

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
    if not df.empty and 'Volume' in df.columns:
        try:
            avg_volume = float(df['Volume'].mean())  # ✅ cast to float to avoid Series comparison error
            avg_volumes.append((ticker, avg_volume))
        except Exception as e:
            st.warning(f"Could not process {ticker}: {e}")

# Sort and select top 3 tickers
top3_tickers = [t[0] for t in sorted(avg_volumes, key=lambda x: x[1], reverse=True)[:3]]

# Retrieve and compile daily volume data for these 3 tickers
volume_data = []
for ticker in top3_tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
    if not df.empty:
        df = df[['Volume']].copy()
        df['Ticker'] = ticker
        df['Date'] = df.index
        volume_data.append(df)

# Plotting
if volume_data:
    combined_df = pd.concat(volume_data, ignore_index=True)
    combined_df = combined_df.pivot(index='Date', columns='Ticker', values='Volume')
    
    st.subheader("📈 Volume Over the Last Month (Top 3 Stocks)")
    st.line_chart(combined_df)
else:
    st.warning("No volume data available to display.")

# -------------------- Tab 2: User Input --------------------
elif page == "User Input":
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

# -------------------- Tab 3: Prediction Output --------------------
elif page == "Prediction Output":
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

# -------------------- Tab 4: Feature Importance --------------------
elif page == "Feature Importance":
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
