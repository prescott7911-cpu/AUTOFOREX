"""
XAU/USD Trend-Following Backtest
Strategy: EMA crossover + pullback + engulfing candle
Exits: 3-way split at previous day liquidity levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# CONFIGURATION — edit these as needed
# ─────────────────────────────────────────

DATA_FILE = "XAUUSD_M15.csv"   # path to your CSV file
ACCOUNT_SIZE = 10000            # starting account in USD ($)
RISK_PCT = 0.025                # 2.5% risk per trade
DAILY_DRAWDOWN_LIMIT = 0.10     # stop trading day if down 10%
NEWS_BUFFER_MINS = 60           # skip trades within 60 mins of news

# Position split (must sum to 1.0)
TP1_PCT = 0.50   # 50% of position closed at TP1
TP2_PCT = 0.30   # 30% at TP2
TP3_PCT = 0.20   # 20% at TP3

# Liquidity sweep offset (pips beyond the level)
SWEEP_PIPS = 2
PIP = 0.1   # 1 pip for XAU/USD = $0.10

# EMA periods
FAST_EMA = 20
SLOW_EMA = 50

# H4 resample label
H4 = "4h"
M15 = "15min"


# ─────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────

def load_data(filepath):
    """
    Expects a CSV with columns: datetime, open, high, low, close
    datetime format: YYYY-MM-DD HH:MM:SS
    Adjust column names below to match your data source.
    """
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.rename(columns={
        "datetime": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    })
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close"]]
    return df


def resample_to_h4(m15_df):
    h4 = m15_df.resample(H4).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()
    return h4


def add_emas(df, fast=FAST_EMA, slow=SLOW_EMA):
    df[f"ema{fast}"] = df["close"].ewm(span=fast, adjust=False).mean()
    df[f"ema{slow}"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


# ─────────────────────────────────────────
# 2. PREVIOUS DAY HIGH / LOW
# ─────────────────────────────────────────

def get_prev_day_levels(m15_df):
    """Returns a series of (prev_high, prev_low) indexed by date."""
    daily = m15_df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    return daily[["prev_high", "prev_low"]]


# ─────────────────────────────────────────
# 3. SIGNAL DETECTION (M15)
# ─────────────────────────────────────────

def is_bullish_engulfing(df, i):
    """Candle i fully engulfs candle i-1 (body-to-body)."""
    prev_open = df["open"].iloc[i - 1]
    prev_close = df["close"].iloc[i - 1]
    curr_open = df["open"].iloc[i]
    curr_close = df["close"].iloc[i]
    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    engulfs = curr_close > prev_open and curr_open < prev_close
    return prev_bearish and curr_bullish and engulfs


def is_bearish_engulfing(df, i):
    prev_open = df["open"].iloc[i - 1]
    prev_close = df["close"].iloc[i - 1]
    curr_open = df["open"].iloc[i]
    curr_close = df["close"].iloc[i]
    prev_bullish = prev_close > prev_open
    curr_bearish = curr_close < curr_open
    engulfs = curr_close < prev_open and curr_open > prev_close
    return prev_bullish and curr_bearish and engulfs


def get_h4_trend(h4_df, timestamp):
    """Return 'bull', 'bear', or None based on H4 EMA alignment."""
    h4_before = h4_df[h4_df.index <= timestamp]
    if h4_before.empty:
        return None
    last = h4_before.iloc[-1]
    if last[f"ema{FAST_EMA}"] > last[f"ema{SLOW_EMA}"]:
        return "bull"
    elif last[f"ema{FAST_EMA}"] < last[f"ema{SLOW_EMA}"]:
        return "bear"
    return None


def price_near_ema(df, i, trend, tolerance_pct=0.002):
    """Check if price has pulled back to within tolerance of the fast EMA."""
    price = df["close"].iloc[i]
    ema = df[f"ema{FAST_EMA}"].iloc[i]
    distance = abs(price - ema) / ema
    if trend == "bull":
        return price >= ema * (1 - tolerance_pct) and price <= ema * (1 + tolerance_pct * 3)
    elif trend == "bear":
        return price <= ema * (1 + tolerance_pct) and price >= ema * (1 - tolerance_pct * 3)
    return False


# ─────────────────────────────────────────
# 4. TP LEVEL SELECTION
# ─────────────────────────────────────────

def select_tp_levels(entry_price, direction, prev_high, prev_low):
    """
    Pick 3 TP levels just beyond previous day liquidity.
    Returns list of 3 prices, or None if no valid levels found.
    """
    sweep = SWEEP_PIPS * PIP

    if direction == "buy":
        levels = []
        if prev_high > entry_price:
            levels.append(prev_high + sweep)
        if prev_low > entry_price and prev_low not in levels:
            levels.append(prev_low + sweep)
        levels = sorted(set(levels))
        if len(levels) < 1:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] + (levels[0] - entry_price) * 0.5)
        return levels[:3]

    elif direction == "sell":
        levels = []
        if prev_low < entry_price:
            levels.append(prev_low - sweep)
        if prev_high < entry_price and prev_high not in levels:
            levels.append(prev_high - sweep)
        levels = sorted(set(levels), reverse=True)
        if len(levels) < 1:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] - (entry_price - levels[0]) * 0.5)
        return levels[:3]

    return None


# ─────────────────────────────────────────
# 5. POSITION SIZING
# ─────────────────────────────────────────

def calc_position_size(account, sl_pips):
    """Risk 2.5% of account. Returns lot size in units."""
    risk_amount = account * RISK_PCT
    pip_value = PIP  # per unit per pip for XAU/USD (approx)
    units = risk_amount / (sl_pips * pip_value)
    return units


# ─────────────────────────────────────────
# 6. MAIN BACKTEST LOOP
# ─────────────────────────────────────────

def run_backtest(m15_df, h4_df, prev_day_levels):
    account = ACCOUNT_SIZE
    trades = []
    daily_start_equity = {}

    for i in range(SLOW_EMA + 1, len(m15_df)):
        timestamp = m15_df.index[i]
        date = timestamp.date()

        if date not in daily_start_equity:
            daily_start_equity[date] = account

        day_loss = (account - daily_start_equity[date]) / daily_start_equity[date]
        if day_loss <= -DAILY_DRAWDOWN_LIMIT:
            continue

        trend = get_h4_trend(h4_df, timestamp)
        if trend is None:
            continue

        if not price_near_ema(m15_df, i, trend):
            continue

        signal = None
        if trend == "bull" and is_bullish_engulfing(m15_df, i):
            signal = "buy"
        elif trend == "bear" and is_bearish_engulfing(m15_df, i):
            signal = "sell"

        if signal is None:
            continue

        close = m15_df["close"].iloc[i]
        ema20 = m15_df[f"ema{FAST_EMA}"].iloc[i]
        ema50 = m15_df[f"ema{SLOW_EMA}"].iloc[i]
        if signal == "buy" and not (close > ema20 > ema50):
            continue
        if signal == "sell" and not (close < ema20 < ema50):
            continue

        prev = prev_day_levels[prev_day_levels.index.date == date]
        if prev.empty:
            continue
        prev_high = prev["prev_high"].iloc[0]
        prev_low = prev["prev_low"].iloc[0]
        if pd.isna(prev_high) or pd.isna(prev_low):
            continue

        entry_price = m15_df["close"].iloc[i]

        if signal == "buy":
            sl_price = m15_df["low"].iloc[i - 1]
            sl_pips = (entry_price - sl_price) / PIP
        else:
            sl_price = m15_df["high"].iloc[i - 1]
            sl_pips = (sl_price - entry_price) / PIP

        if sl_pips <= 0:
            continue

        tps = select_tp_levels(entry_price, signal, prev_high, prev_low)
        if tps is None:
            continue

        tp1, tp2, tp3 = tps

        total_units = calc_position_size(account, sl_pips)
        units = {
            "tp1": total_units * TP1_PCT,
            "tp2": total_units * TP2_PCT,
            "tp3": total_units * TP3_PCT,
        }

        trade_result = simulate_trade(
            m15_df, i + 1, signal,
            entry_price, sl_price, tp1, tp2, tp3,
            units, account
        )

        pnl = trade_result["pnl"]
        account += pnl

        trades.append({
            "entry_time": timestamp,
            "exit_time": trade_result["exit_time"],
            "direction": signal,
            "entry": entry_price,
            "sl": sl_price,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "sl_pips": sl_pips,
            "pnl": pnl,
            "account": account,
            "outcome": trade_result["outcome"],
        })

    return pd.DataFrame(trades)


def simulate_trade(df, start_i, direction, entry, sl, tp1, tp2, tp3, units, account):
    """Walk forward candle by candle until all TPs or SL is hit."""
    remaining = dict(units)
    pnl = 0.0
    sl_moved = False
    current_sl = sl
    outcome_parts = []

    for i in range(start_i, len(df)):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        time = df.index[i]

        if direction == "buy":
            if low <= current_sl:
                for key, u in remaining.items():
                    pnl -= u * (entry - current_sl)
                outcome_parts.append("SL")
                remaining = {}
                break

            if "tp1" in remaining and high >= tp1:
                pnl += remaining["tp1"] * (tp1 - entry)
                outcome_parts.append("TP1")
                del remaining["tp1"]
                if not sl_moved:
                    current_sl = entry
                    sl_moved = True

            if "tp2" in remaining and high >= tp2:
                pnl += remaining["tp2"] * (tp2 - entry)
                outcome_parts.append("TP2")
                del remaining["tp2"]

            if "tp3" in remaining and high >= tp3:
                pnl += remaining["tp3"] * (tp3 - entry)
                outcome_parts.append("TP3")
                del remaining["tp3"]

        elif direction == "sell":
            if high >= current_sl:
                for key, u in remaining.items():
                    pnl -= u * (current_sl - entry)
                outcome_parts.append("SL")
                remaining = {}
                break

            if "tp1" in remaining and low <= tp1:
                pnl += remaining["tp1"] * (entry - tp1)
                outcome_parts.append("TP1")
                del remaining["tp1"]
                if not sl_moved:
                    current_sl = entry
                    sl_moved = True

            if "tp2" in remaining and low <= tp2:
                pnl += remaining["tp2"] * (entry - tp2)
                outcome_parts.append("TP2")
                del remaining["tp2"]

            if "tp3" in remaining and low <= tp3:
                pnl += remaining["tp3"] * (entry - tp3)
                outcome_parts.append("TP3")
                del remaining["tp3"]

        if not remaining:
            return {"pnl": pnl, "exit_time": time, "outcome": "+".join(outcome_parts)}

        if i - start_i > 5 * 96:
            for key, u in remaining.items():
                close = df["close"].iloc[i]
                if direction == "buy":
                    pnl += u * (close - entry)
                else:
                    pnl += u * (entry - close)
            outcome_parts.append("TIMEOUT")
            return {"pnl": pnl, "exit_time": time, "outcome": "+".join(outcome_parts)}

    return {"pnl": pnl, "exit_time": df.index[-1], "outcome": "+".join(outcome_parts) or "OPEN"}


# ─────────────────────────────────────────
# 7. RESULTS & REPORTING
# ─────────────────────────────────────────

def print_results(trades_df):
    if trades_df.empty:
        print("No trades found. Check your data file and column names.")
        return

    total = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / total * 100
    total_pnl = trades_df["pnl"].sum()
    avg_win = winners["pnl"].mean() if not winners.empty else 0
    avg_loss = losers["pnl"].mean() if not losers.empty else 0
    profit_factor = (winners["pnl"].sum() / abs(losers["pnl"].sum())) if not losers.empty else float("inf")
    max_dd = calc_max_drawdown(trades_df["account"])
    final_account = trades_df["account"].iloc[-1]
    return_pct = (final_account - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    print("\n" + "="*50)
    print("  XAU/USD BACKTEST RESULTS")
    print("="*50)
    print(f"  Period:          {trades_df['entry_time'].min().date()} → {trades_df['entry_time'].max().date()}")
    print(f"  Total trades:    {total}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Profit factor:   {profit_factor:.2f}")
    print(f"  Avg win:         ${avg_win:.2f}")
    print(f"  Avg loss:        ${avg_loss:.2f}")
    print(f"  Total P&L:       ${total_pnl:.2f}")
    print(f"  Return:          {return_pct:.1f}%")
    print(f"  Max drawdown:    {max_dd:.1f}%")
    print(f"  Final account:   ${final_account:.2f}")
    print("="*50)

    print("\n  Outcome breakdown:")
    print(trades_df["outcome"].value_counts().to_string())
    print()


def calc_max_drawdown(equity_series):
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    return drawdown.min()


def plot_results(trades_df):
    if trades_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("XAU/USD Backtest — Strategy Results", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(trades_df["entry_time"], trades_df["account"], color="#2563eb", linewidth=1.5)
    ax1.axhline(ACCOUNT_SIZE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("Equity curve")
    ax1.set_ylabel("Account ($)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    colors = ["#16a34a" if p > 0 else "#dc2626" for p in trades_df["pnl"]]
    ax2.bar(range(len(trades_df)), trades_df["pnl"], color=colors, width=0.8)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("P&L per trade")
    ax2.set_ylabel("P&L ($)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    equity = trades_df["account"]
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    ax3.fill_between(trades_df["entry_time"], dd, 0, color="#dc2626", alpha=0.3)
    ax3.plot(trades_df["entry_time"], dd, color="#dc2626", linewidth=1)
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("Drawdown (%)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("  Chart saved to: backtest_results.png")
    plt.show()


# ─────────────────────────────────────────
# 8. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading data from: {DATA_FILE}")

    if not Path(DATA_FILE).exists():
        print(f"\nERROR: File '{DATA_FILE}' not found.")
        print("Download XAU/USD M15 OHLC data and save it as 'XAUUSD_M15.csv'")
        print("Expected columns: datetime, open, high, low, close")
        print("\nFree data sources:")
        print("  • https://www.histdata.com  (select XAUUSD, M1, then resample)")
        print("  • https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed/")
        exit(1)

    m15_df = load_data(DATA_FILE)
    print(f"Loaded {len(m15_df):,} M15 candles: {m15_df.index[0]} → {m15_df.index[-1]}")

    m15_df = add_emas(m15_df)
    h4_df = resample_to_h4(m15_df)
    h4_df = add_emas(h4_df)
    prev_day_levels = get_prev_day_levels(m15_df)

    print("Running backtest...")
    trades = run_backtest(m15_df, h4_df, prev_day_levels)

    print_results(trades)
    trades.to_csv("backtest_trades.csv", index=False)
    print("  Trade log saved to: backtest_trades.csv")

    plot_results(trades)
