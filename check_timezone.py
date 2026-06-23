"""
Run this first to check what timezone your data is in.
It prints the first 20 timestamps and the hours that appear most frequently,
so you can see what offset to apply.
"""
import pandas as pd
import sys

DATA_FILE = "XAUUSD_M15.csv"

try:
    df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "time"})
    df = df.set_index("time").sort_index()
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

print("\n-- First 20 timestamps ----------------------")
print(df.index[:20].tolist())

print("\n-- Hour distribution (all candles) ----------")
hour_counts = df.index.hour
dist = pd.Series(hour_counts).value_counts().sort_index()
for hour, count in dist.items():
    bar = "X" * (count // 10)
    print(f"  {hour:02d}:00  {count:>5}  {bar}")

print("\n-- What to look for -------------------------")
print("  London open  = 07:00 GMT")
print("  NY open      = 13:00 GMT")
print("  If your data shows high activity at e.g. 02:00-05:00,")
print("  your data is likely in US Eastern time (GMT-5).")
print("  Adjust SESSION_WINDOWS_GMT in backtest_v3.py accordingly.")
