"""
XAU/USD Trend-Following Backtest -- v3
New filters vs v2:
  1. H4 confluence -- entry only within 50 pips of H4 20 EMA
  2. Pullback must hold 2 candles before engulfing forms
  3. Session filter -- London (07:00-10:00 GMT) and New York (13:00-16:00 GMT) only
  4. ATR volatility filter -- M15 ATR(14) must be above its 20-period average
  5. Daily range breakout filter -- only trade in direction of yesterday's range break
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

DATA_FILE = "XAUUSD_M15.csv"
ACCOUNT_SIZE = 10000
RISK_PCT = 0.01
DAILY_DRAWDOWN_LIMIT = 0.10

TP1_PCT = 0.50
TP2_PCT = 0.30
TP3_PCT = 0.20

SWEEP_PIPS = 2
PIP = 0.1

FAST_EMA = 20
SLOW_EMA = 50
MIN_RR = 1.5

# NEW v3 settings
H4_EMA_PROXIMITY_PIPS = 50        # Filter 1: max pips from H4 20 EMA at entry
SESSION_WINDOWS_GMT = [            # Filter 3: allowed trading hours (UTC/GMT)
    (7, 12),                       # London open (extended)
    (13, 17),                      # New York open (extended)
]
ATR_PERIOD = 14                    # Filter 4: ATR period
ATR_MA_PERIOD = 20                 # Filter 4: ATR moving average period

H4 = "4h"


# -----------------------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------------------

def load_data(filepath):
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
    return m15_df.resample(H4).agg({
        "open": "first", "high": "max",
        "low": "min",    "close": "last"
    }).dropna()


def add_emas(df, fast=FAST_EMA, slow=SLOW_EMA):
    df[f"ema{fast}"] = df["close"].ewm(span=fast, adjust=False).mean()
    df[f"ema{slow}"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def add_atr(df, period=ATR_PERIOD, ma_period=ATR_MA_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"]    = tr.ewm(span=period, adjust=False).mean()
    df["atr_ma"] = df["atr"].rolling(ma_period).mean()
    return df


# -----------------------------------------
# 2. PREVIOUS DAY HIGH / LOW
# -----------------------------------------

def get_prev_day_levels(m15_df):
    daily = m15_df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"]  = daily["low"].shift(1)
    return daily[["prev_high", "prev_low"]]


# -----------------------------------------
# 3. FILTERS
# -----------------------------------------

def is_in_session(timestamp):
    hour = timestamp.hour
    for start, end in SESSION_WINDOWS_GMT:
        if start <= hour < end:
            return True
    return False


def is_near_h4_ema(h4_df, timestamp, entry_price):
    h4_before = h4_df[h4_df.index <= timestamp]
    if h4_before.empty:
        return False
    h4_ema = h4_before[f"ema{FAST_EMA}"].iloc[-1]
    distance_pips = abs(entry_price - h4_ema) / PIP
    return distance_pips <= H4_EMA_PROXIMITY_PIPS


def is_atr_expanding(df, i):
    if pd.isna(df["atr_ma"].iloc[i]):
        return False
    return df["atr"].iloc[i] > df["atr_ma"].iloc[i]


def daily_breakout_confirmed(direction, entry_price, prev_high, prev_low):
    if direction == "buy":
        return entry_price > prev_high
    elif direction == "sell":
        return entry_price < prev_low
    return False


def get_h4_trend(h4_df, timestamp):
    h4_before = h4_df[h4_df.index <= timestamp]
    if h4_before.empty:
        return None
    last = h4_before.iloc[-1]
    if last[f"ema{FAST_EMA}"] > last[f"ema{SLOW_EMA}"]:
        return "bull"
    elif last[f"ema{FAST_EMA}"] < last[f"ema{SLOW_EMA}"]:
        return "bear"
    return None


# -----------------------------------------
# 4. SIGNAL DETECTION
# -----------------------------------------

def is_bullish_engulfing(df, i):
    p_o, p_c = df["open"].iloc[i-1], df["close"].iloc[i-1]
    c_o, c_c = df["open"].iloc[i],   df["close"].iloc[i]
    return (p_c < p_o and c_c > c_o and c_c > p_o and c_o < p_c)


def is_bearish_engulfing(df, i):
    p_o, p_c = df["open"].iloc[i-1], df["close"].iloc[i-1]
    c_o, c_c = df["open"].iloc[i],   df["close"].iloc[i]
    return (p_c > p_o and c_c < c_o and c_c < p_o and c_o > p_c)


def pullback_held_two_candles(df, i, trend, tolerance=0.002):
    if i < 3:
        return False
    for lookback in [2, 1]:
        idx   = i - lookback
        price = df["close"].iloc[idx]
        ema   = df[f"ema{FAST_EMA}"].iloc[idx]
        dist  = abs(price - ema) / ema
        if dist > tolerance:
            return False
        if trend == "bull" and price > ema * (1 + tolerance * 2):
            return False
        if trend == "bear" and price < ema * (1 - tolerance * 2):
            return False
    return True


# -----------------------------------------
# 5. TP LEVEL SELECTION
# -----------------------------------------

def select_tp_levels(entry_price, direction, prev_high, prev_low, sl_pips):
    sweep = SWEEP_PIPS * PIP
    min_tp1_dist = sl_pips * PIP * MIN_RR

    if direction == "buy":
        levels = sorted(set(
            [l + sweep for l in [prev_high, prev_low] if l > entry_price]
        ))
        if not levels or (levels[0] - entry_price) < min_tp1_dist:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] + (levels[0] - entry_price) * 0.5)
        return levels[:3]

    elif direction == "sell":
        levels = sorted(set(
            [l - sweep for l in [prev_low, prev_high] if l < entry_price]
        ), reverse=True)
        if not levels or (entry_price - levels[0]) < min_tp1_dist:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] - (entry_price - levels[0]) * 0.5)
        return levels[:3]

    return None


# -----------------------------------------
# 6. POSITION SIZING
# -----------------------------------------

def calc_position_size(account, sl_pips):
    return (account * RISK_PCT) / (sl_pips * PIP)


# -----------------------------------------
# 7. MAIN BACKTEST LOOP
# -----------------------------------------

def run_backtest(m15_df, h4_df, prev_day_levels):
    account = ACCOUNT_SIZE
    trades  = []
    daily_start_equity = {}

    skipped = {
        "session":   0,
        "h4_prox":   0,
        "pullback":  0,
        "atr":       0,
        "breakout":  0,
        "rr":        0,
        "no_signal": 0,
    }

    for i in range(max(SLOW_EMA, ATR_MA_PERIOD) + 3, len(m15_df)):
        timestamp = m15_df.index[i]
        date      = timestamp.date()

        if date not in daily_start_equity:
            daily_start_equity[date] = account

        if (account - daily_start_equity[date]) / daily_start_equity[date] <= -DAILY_DRAWDOWN_LIMIT:
            continue

        if not is_in_session(timestamp):
            skipped["session"] += 1
            continue

        trend = get_h4_trend(h4_df, timestamp)
        if trend is None:
            continue

        if not is_atr_expanding(m15_df, i):
            skipped["atr"] += 1
            continue

        if not pullback_held_two_candles(m15_df, i, trend):
            skipped["pullback"] += 1
            continue

        signal = None
        if trend == "bull" and is_bullish_engulfing(m15_df, i):
            signal = "buy"
        elif trend == "bear" and is_bearish_engulfing(m15_df, i):
            signal = "sell"

        if signal is None:
            skipped["no_signal"] += 1
            continue

        close = m15_df["close"].iloc[i]
        ema20 = m15_df[f"ema{FAST_EMA}"].iloc[i]
        ema50 = m15_df[f"ema{SLOW_EMA}"].iloc[i]
        if signal == "buy"  and not (close > ema20 > ema50): continue
        if signal == "sell" and not (close < ema20 < ema50): continue

        if not is_near_h4_ema(h4_df, timestamp, close):
            skipped["h4_prox"] += 1
            continue

        prev = prev_day_levels[prev_day_levels.index.date == date]
        if prev.empty: continue
        prev_high = prev["prev_high"].iloc[0]
        prev_low  = prev["prev_low"].iloc[0]
        if pd.isna(prev_high) or pd.isna(prev_low): continue

        if not daily_breakout_confirmed(signal, close, prev_high, prev_low):
            skipped["breakout"] += 1
            continue

        entry_price = close

        if signal == "buy":
            sl_price = m15_df["low"].iloc[i - 1]
            sl_pips  = (entry_price - sl_price) / PIP
        else:
            sl_price = m15_df["high"].iloc[i - 1]
            sl_pips  = (sl_price - entry_price) / PIP

        if sl_pips <= 0: continue

        tps = select_tp_levels(entry_price, signal, prev_high, prev_low, sl_pips)
        if tps is None:
            skipped["rr"] += 1
            continue

        tp1, tp2, tp3 = tps
        total_units   = calc_position_size(account, sl_pips)
        units = {
            "tp1": total_units * TP1_PCT,
            "tp2": total_units * TP2_PCT,
            "tp3": total_units * TP3_PCT,
        }

        result  = simulate_trade(m15_df, i + 1, signal, entry_price, sl_price, tp1, tp2, tp3, units)
        pnl     = result["pnl"]
        account += pnl

        tp1_rr = ((tp1 - entry_price) / (entry_price - sl_price) if signal == "buy"
                  else (entry_price - tp1) / (sl_price - entry_price))

        trades.append({
            "entry_time": timestamp,
            "exit_time":  result["exit_time"],
            "direction":  signal,
            "entry":      entry_price,
            "sl":         sl_price,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "sl_pips":    round(sl_pips, 1),
            "tp1_rr":     round(tp1_rr, 2),
            "pnl":        round(pnl, 2),
            "account":    round(account, 2),
            "outcome":    result["outcome"],
        })

    print("\n  -- Filter skip breakdown ----------------------")
    for key, val in skipped.items():
        print(f"  {key:<14} {val:>5} candles filtered")
    print(f"  {'TRADES TAKEN':<14} {len(trades):>5}")
    print("  -----------------------------------------------")

    return pd.DataFrame(trades)


# -----------------------------------------
# 8. TRADE SIMULATION
# -----------------------------------------

def simulate_trade(df, start_i, direction, entry, sl, tp1, tp2, tp3, units):
    remaining    = dict(units)
    pnl          = 0.0
    sl_moved     = False
    current_sl   = sl
    outcome_parts = []

    for i in range(start_i, len(df)):
        high = df["high"].iloc[i]
        low  = df["low"].iloc[i]
        time = df.index[i]

        if direction == "buy":
            if low <= current_sl:
                for u in remaining.values():
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
                for u in remaining.values():
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
            close = df["close"].iloc[i]
            for u in remaining.values():
                pnl += u * (close - entry) if direction == "buy" else u * (entry - close)
            outcome_parts.append("TIMEOUT")
            return {"pnl": pnl, "exit_time": time, "outcome": "+".join(outcome_parts)}

    return {"pnl": pnl, "exit_time": df.index[-1], "outcome": "+".join(outcome_parts) or "OPEN"}


# -----------------------------------------
# 9. RESULTS & REPORTING
# -----------------------------------------

def print_results(trades_df):
    if trades_df.empty:
        print("\nNo trades generated.")
        print("Try widening H4_EMA_PROXIMITY_PIPS or relaxing pullback tolerance.")
        return

    total   = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers  = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / total * 100
    pf = (winners["pnl"].sum() / abs(losers["pnl"].sum())) if not losers.empty else float("inf")
    max_dd = calc_max_drawdown(trades_df["account"])
    final  = trades_df["account"].iloc[-1]
    ret    = (final - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    print("\n" + "="*54)
    print("  XAU/USD BACKTEST RESULTS -- v3")
    print("="*54)
    print(f"  Period:          {trades_df['entry_time'].min().date()} to {trades_df['entry_time'].max().date()}")
    print(f"  Total trades:    {total}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Avg TP1 R:R:     {trades_df['tp1_rr'].mean():.2f}x")
    print(f"  Avg win:         ${winners['pnl'].mean():.2f}" if not winners.empty else "  Avg win:         --")
    print(f"  Avg loss:        ${losers['pnl'].mean():.2f}"  if not losers.empty  else "  Avg loss:         --")
    print(f"  Total P&L:       ${trades_df['pnl'].sum():.2f}")
    print(f"  Return:          {ret:.1f}%")
    print(f"  Max drawdown:    {max_dd:.1f}%")
    print(f"  Final account:   ${final:.2f}")
    print("="*54)
    print("\n  Outcome breakdown:")
    print(trades_df["outcome"].value_counts().to_string())
    print()
    print("  -- Version comparison -------------------------")
    print("  v1 -- Trades: 55  | WR: 38.2% | PF: 0.96 | DD: -27.1% | Ret: -3.5%")
    print(f"  v3 -- Trades: {total:<3}  | WR: {win_rate:.1f}%  | PF: {pf:.2f}  | DD: {max_dd:.1f}%  | Ret: {ret:.1f}%")
    print("  -----------------------------------------------\n")


def calc_max_drawdown(equity_series):
    peak = equity_series.cummax()
    return ((equity_series - peak) / peak * 100).min()


def plot_results(trades_df):
    if trades_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("XAU/USD Backtest v3 -- With Session + ATR + H4 + Breakout Filters", fontsize=13)

    ax1 = axes[0]
    ax1.plot(trades_df["entry_time"], trades_df["account"], color="#2563eb", linewidth=1.8)
    ax1.axhline(ACCOUNT_SIZE, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
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
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    ax3.fill_between(trades_df["entry_time"], dd, 0, color="#dc2626", alpha=0.3)
    ax3.plot(trades_df["entry_time"], dd, color="#dc2626", linewidth=1)
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("Drawdown (%)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("backtest_results_v3.png", dpi=150, bbox_inches="tight")
    print("  Chart saved to: backtest_results_v3.png")
    plt.show()


# -----------------------------------------
# 10. RUN
# -----------------------------------------

if __name__ == "__main__":
    print(f"\nLoading data from: {DATA_FILE}")

    if not Path(DATA_FILE).exists():
        print(f"\nERROR: '{DATA_FILE}' not found.")
        print("Run download_data.py first to fetch XAU/USD M15 data.")
        exit(1)

    m15_df = load_data(DATA_FILE)
    print(f"Loaded {len(m15_df):,} M15 candles: {m15_df.index[0]} to {m15_df.index[-1]}")

    m15_df = add_emas(m15_df)
    m15_df = add_atr(m15_df)
    h4_df  = resample_to_h4(m15_df)
    h4_df  = add_emas(h4_df)
    prev_day_levels = get_prev_day_levels(m15_df)

    print("Running backtest with v3 filters...")
    trades = run_backtest(m15_df, h4_df, prev_day_levels)

    print_results(trades)
    trades.to_csv("backtest_trades_v3.csv", index=False)
    print("  Trade log saved to: backtest_trades_v3.csv")

    plot_results(trades)
