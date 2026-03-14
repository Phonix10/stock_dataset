import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ---------------------------------
# CONFIG
# ---------------------------------

YEARS = 15
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * YEARS)

# NIFTY 50 Stocks
nifty50 = [
"RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","HINDUNILVR.NS",
"ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
"ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS",
"NESTLEIND.NS","BAJFINANCE.NS","BAJAJFINSV.NS","HCLTECH.NS","WIPRO.NS",
"ONGC.NS","NTPC.NS","POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS",
"COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","GRASIM.NS",
"TECHM.NS","INDUSINDBK.NS","DRREDDY.NS","CIPLA.NS","BRITANNIA.NS",
"EICHERMOT.NS","DIVISLAB.NS","HEROMOTOCO.NS","APOLLOHOSP.NS",
"TATACONSUM.NS","BAJAJ-AUTO.NS","BPCL.NS","UPL.NS","HDFCLIFE.NS",
"SBILIFE.NS","ICICIPRULI.NS","SHREECEM.NS","TATAMOTORS.NS"
]

# Create stock folder
os.makedirs("stock", exist_ok=True)

# ---------------------------------
# MACRO DATA
# ---------------------------------

print("Downloading macro data...")

oil = yf.download("CL=F", start=start_date, end=end_date)
usd = yf.download("INR=X", start=start_date, end=end_date)
sp500 = yf.download("^GSPC", start=start_date, end=end_date)

oil.columns = oil.columns.get_level_values(0)
usd.columns = usd.columns.get_level_values(0)
sp500.columns = sp500.columns.get_level_values(0)

oil = oil[["Close"]].rename(columns={"Close": "OilPrice"})
usd = usd[["Close"]].rename(columns={"Close": "USDINR"})
sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})

oil.reset_index(inplace=True)
usd.reset_index(inplace=True)
sp500.reset_index(inplace=True)

# ---------------------------------
# PROCESS EACH STOCK
# ---------------------------------

for ticker in nifty50:

    print("Processing:", ticker)

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print("No data:", ticker)
        continue

    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)

    # Moving averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # OBV
    obv = [0]

    for i in range(1, len(df)):

        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])

        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])

        else:
            obv.append(obv[-1])

    df["OBV"] = obv

    # Fibonacci
    max_price = df["High"].max()
    min_price = df["Low"].min()

    diff = max_price - min_price

    df["Fib23"] = max_price - 0.236 * diff
    df["Fib38"] = max_price - 0.382 * diff
    df["Fib50"] = max_price - 0.5 * diff
    df["Fib61"] = max_price - 0.618 * diff

    # Merge macro data
    df = df.merge(oil, on="Date", how="left")
    df = df.merge(usd, on="Date", how="left")
    df = df.merge(sp500, on="Date", how="left")

    # Keep only required columns
    df = df[
        [
            "Date","Close","High","Low","Open","Volume",
            "MA20","MA50","EMA20","EMA50","RSI","OBV",
            "Fib23","Fib38","Fib50","Fib61",
            "OilPrice","USDINR","SP500"
        ]
    ]

    df.dropna(inplace=True)

    # Save file
    stock_name = ticker.replace(".NS","")

    file_path = f"stock/{stock_name}.xlsx"

    df.to_excel(file_path, index=False)

    print("Saved:", file_path)

print("All stocks processed.")