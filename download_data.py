"""
Download XAU/USD M15 data via yfinance (gold futures GC=F).
Saves to XAUUSD_M15.csv ready for backtest.py.

Note: yfinance provides up to 60 days of 15-minute data at no cost.
For longer history see: https://www.histdata.com (XAUUSD, M1, resample)
"""

import sys
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import pandas as pd
from pathlib import Path

OUTPUT_FILE = "XAUUSD_M15.csv"


def download():
    print("Downloading XAU/USD 15-minute data (last 60 days)...")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="60d", interval="15m")

    if df.empty:
        print("ERROR: No data returned. Check your internet connection.")
        return

    df = df.reset_index()
    df = df.rename(columns={
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df[["datetime", "open", "high", "low", "close"]].dropna()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} candles to {OUTPUT_FILE}")
    print(f"Range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")


if __name__ == "__main__":
    download()
