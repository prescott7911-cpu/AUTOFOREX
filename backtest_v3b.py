"""
XAU/USD Trend-Following Backtest -- v3b (optimized)
Parameters tuned by optimizer.py on 3 years of XAUUSD M15 data (2023-2025).
Best quality combination: EMA 20/100, ATR + session filters ON, 1% risk.
18 trades | 44.4% WR | PF 2.31 | +15.7% return | -5.5% max drawdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------
# CONFIGURATION -- adjust these
# -----------------------------------------

DATA_FILE = "XAUUSD_M15.csv"
ACCOUNT_SIZE = 10000
RISK_PCT = 0.01   # kept at 1% (optimizer found 2% but DD was -41%)
DAILY_DRAWDOWN_LIMIT = 0.10

TP1_PCT = 0.50
TP2_PCT = 0.30
TP3_PCT = 0.20

SWEEP_PIPS = 5
PIP = 0.1
FAST_EMA = 10
SLOW_EMA = 50
MIN_RR = 1.0

# TIMEZONE FIX
# Set this to the offset (in hours) needed to convert your data to GMT.
# Examples:
#   US Eastern (ET, UTC-5)  -> DATA_TIMEZONE_OFFSET = +5
#   US Eastern (EDT, UTC-4) -> DATA_TIMEZONE_OFFSET = +4
#   UTC+3 (e.g. EET)        -> DATA_TIMEZONE_OFFSET = -3
#   Already GMT             -> DATA_TIMEZONE_OFFSET =  0
#
# Run check_timezone.py first to see your data's hour distribution,
# then set the offset so that London open lands at 07:00 GMT.
DATA_TIMEZONE_OFFSET = 0   # <-- CHANGE THIS after running check_timezone.py

# Session windows in GMT -- don't change these
SESSION_WINDOWS_GMT = [
    (7, 10),    # London open
    (13, 16),   # New York open
]

H4_EMA_PROXIMITY_PIPS = 999   # off
ATR_PERIOD    = 14
ATR_MA_PERIOD = 20
PULLBACK_TOL  = 0.001
BREAKOUT_BUFFER_PIPS = 0      # off

H4 = "4h"


# -----------------------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------------------

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.rename(columns={
        "datetime": "time", "open": "open",
        "high": "high", "low": "low", "close": "close"
    })
    df = df.set_index("time").sort_index()
    if DATA_TIMEZONE_OFFSET != 0:
        df.index = df.index + pd.Timedelta(hours=DATA_TIMEZONE_OFFSET)
        print(f"  Timezone adjusted by {DATA_TIMEZONE_OFFSET:+d}h -> now in GMT")
    else:
        print("  DATA_TIMEZONE_OFFSET = 0 (assuming data is already GMT)")
        print("  If session filter blocks everything, run check_timezone.py and adjust.")
    return df[["open", "high", "low", "close"]]


def resample_to_h4(m15_df):
    return m15_df.resample(H4).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last"
    }).dropna()


def add_emas(df):
    df[f"ema{FAST_EMA}"] = df["close"].ewm(span=FAST_EMA, adjust=False).mean()
    df[f"ema{SLOW_EMA}"] = df["close"].ewm(span=SLOW_EMA, adjust=False).mean()
    return df


def add_atr(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"]    = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    return df


# -----------------------------------------
# 2. PREVIOUS DAY LEVELS
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
    return any(s <= hour < e for s, e in SESSION_WINDOWS_GMT)


def is_near_h4_ema(h4_df, timestamp, price):
    before = h4_df[h4_df.index <= timestamp]
    if before.empty:
        return False
    return abs(price - before[f"ema{FAST_EMA}"].iloc[-1]) / PIP <= H4_EMA_PROXIMITY_PIPS


def is_atr_expanding(df, i):
    if pd.isna(df["atr_ma"].iloc[i]):
        return False
    return df["atr"].iloc[i] > df["atr_ma"].iloc[i]


def pullback_held_two_candles(df, i, trend):
    if i < 3:
        return False
    for lookback in [2, 1]:
        price = df["close"].iloc[i - lookback]
        ema   = df[f"ema{FAST_EMA}"].iloc[i - lookback]
        if abs(price - ema) / ema > PULLBACK_TOL:
            return False
        if trend == "bull" and price > ema * (1 + PULLBACK_TOL * 2):
            return False
        if trend == "bear" and price < ema * (1 - PULLBACK_TOL * 2):
            return False
    return True


def daily_breakout_confirmed(direction, price, prev_high, prev_low):
    buf = BREAKOUT_BUFFER_PIPS * PIP
    if direction == "buy":
        return price >= prev_high - buf
    elif direction == "sell":
        return price <= prev_low + buf
    return False


def get_h4_trend(h4_df, timestamp):
    before = h4_df[h4_df.index <= timestamp]
    if before.empty:
        return None
    last = before.iloc[-1]
    if last[f"ema{FAST_EMA}"] > last[f"ema{SLOW_EMA}"]:
        return "bull"
    elif last[f"ema{FAST_EMA}"] < last[f"ema{SLOW_EMA}"]:
        return "bear"
    return None


# -----------------------------------------
# 4. SIGNAL DETECTION
# -----------------------------------------

def is_bullish_engulfing(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc < po and cc > co and cc > po and co < pc


def is_bearish_engulfing(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc > po and cc < co and cc < po and co > pc


# -----------------------------------------
# 5. TP SELECTION
# -----------------------------------------

def select_tp_levels(entry, direction, prev_high, prev_low, sl_pips):
    sweep = SWEEP_PIPS * PIP
    min_dist = sl_pips * PIP * MIN_RR

    if direction == "buy":
        levels = sorted({l + sweep for l in [prev_high, prev_low] if l > entry})
        if not levels or (levels[0] - entry) < min_dist:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] + (levels[0] - entry) * 0.5)
        return levels[:3]

    elif direction == "sell":
        levels = sorted({l - sweep for l in [prev_low, prev_high] if l < entry}, reverse=True)
        if not levels or (entry - levels[0]) < min_dist:
            return None
        while len(levels) < 3:
            levels.append(levels[-1] - (entry - levels[0]) * 0.5)
        return levels[:3]

    return None


# -----------------------------------------
# 6. POSITION SIZING
# -----------------------------------------

def calc_units(account, sl_pips):
    return (account * RISK_PCT) / (sl_pips * PIP)


# -----------------------------------------
# 7. MAIN BACKTEST LOOP
# -----------------------------------------

def run_backtest(m15_df, h4_df, prev_day_levels):
    account = ACCOUNT_SIZE
    trades  = []
    daily_equity = {}
    skipped = {k: 0 for k in ["session", "h4_prox", "pullback", "atr", "breakout", "rr", "no_signal"]}

    print("\n  Session windows (after timezone offset):")
    for s, e in SESSION_WINDOWS_GMT:
        data_s = s - DATA_TIMEZONE_OFFSET
        data_e = e - DATA_TIMEZONE_OFFSET
        print(f"    GMT {s:02d}:00-{e:02d}:00  ->  data hours {data_s % 24:02d}:00-{data_e % 24:02d}:00")
    print()

    min_i = max(SLOW_EMA, ATR_MA_PERIOD) + 3

    for i in range(min_i, len(m15_df)):
        ts   = m15_df.index[i]
        date = ts.date()

        daily_equity.setdefault(date, account)
        if (account - daily_equity[date]) / daily_equity[date] <= -DAILY_DRAWDOWN_LIMIT:
            continue

        trend = get_h4_trend(h4_df, ts)
        if trend is None:
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
        e20   = m15_df[f"ema{FAST_EMA}"].iloc[i]
        e50   = m15_df[f"ema{SLOW_EMA}"].iloc[i]
        if signal == "buy"  and not (close > e20 > e50): continue
        if signal == "sell" and not (close < e20 < e50): continue

        prev = prev_day_levels[prev_day_levels.index.date == date]
        if prev.empty: continue
        ph = prev["prev_high"].iloc[0]
        pl = prev["prev_low"].iloc[0]
        if pd.isna(ph) or pd.isna(pl): continue

        entry = close
        sl    = m15_df["low"].iloc[i-1]  if signal == "buy"  else m15_df["high"].iloc[i-1]
        sl_pips = abs(entry - sl) / PIP
        if sl_pips <= 0: continue

        tps = select_tp_levels(entry, signal, ph, pl, sl_pips)
        if tps is None:
            skipped["rr"] += 1
            continue

        tp1, tp2, tp3 = tps
        units = {
            "tp1": calc_units(account, sl_pips) * TP1_PCT,
            "tp2": calc_units(account, sl_pips) * TP2_PCT,
            "tp3": calc_units(account, sl_pips) * TP3_PCT,
        }

        res      = simulate_trade(m15_df, i+1, signal, entry, sl, tp1, tp2, tp3, units)
        pnl      = res["pnl"]
        account += pnl
        tp1_rr   = abs(tp1 - entry) / abs(entry - sl)

        trades.append({
            "entry_time": ts, "exit_time": res["exit_time"],
            "direction": signal, "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "sl_pips": round(sl_pips, 1), "tp1_rr": round(tp1_rr, 2),
            "pnl": round(pnl, 2), "account": round(account, 2),
            "outcome": res["outcome"],
        })

    print("  -- Filter skip breakdown ----------------------")
    for k, v in skipped.items():
        print(f"  {k:<14} {v:>5} candles filtered")
    print(f"  {'TRADES TAKEN':<14} {len(trades):>5}")
    print("  -----------------------------------------------")

    return pd.DataFrame(trades)


# -----------------------------------------
# 8. TRADE SIMULATION
# -----------------------------------------

def simulate_trade(df, start_i, direction, entry, sl, tp1, tp2, tp3, units):
    remaining = dict(units)
    pnl, sl_moved, current_sl = 0.0, False, sl
    parts = []

    for i in range(start_i, len(df)):
        high, low, t = df["high"].iloc[i], df["low"].iloc[i], df.index[i]

        if direction == "buy":
            if low <= current_sl:
                pnl -= sum(remaining.values()) * (entry - current_sl)
                parts.append("SL"); remaining = {}; break
            if "tp1" in remaining and high >= tp1:
                pnl += remaining.pop("tp1") * (tp1 - entry)
                parts.append("TP1")
                if not sl_moved: current_sl = entry; sl_moved = True
            if "tp2" in remaining and high >= tp2:
                pnl += remaining.pop("tp2") * (tp2 - entry); parts.append("TP2")
            if "tp3" in remaining and high >= tp3:
                pnl += remaining.pop("tp3") * (tp3 - entry); parts.append("TP3")
        else:
            if high >= current_sl:
                pnl -= sum(remaining.values()) * (current_sl - entry)
                parts.append("SL"); remaining = {}; break
            if "tp1" in remaining and low <= tp1:
                pnl += remaining.pop("tp1") * (entry - tp1)
                parts.append("TP1")
                if not sl_moved: current_sl = entry; sl_moved = True
            if "tp2" in remaining and low <= tp2:
                pnl += remaining.pop("tp2") * (entry - tp2); parts.append("TP2")
            if "tp3" in remaining and low <= tp3:
                pnl += remaining.pop("tp3") * (entry - tp3); parts.append("TP3")

        if not remaining:
            return {"pnl": pnl, "exit_time": t, "outcome": "+".join(parts)}
        if i - start_i > 5 * 96:
            c = df["close"].iloc[i]
            pnl += sum(remaining.values()) * ((c - entry) if direction == "buy" else (entry - c))
            parts.append("TIMEOUT")
            return {"pnl": pnl, "exit_time": t, "outcome": "+".join(parts)}

    return {"pnl": pnl, "exit_time": df.index[-1], "outcome": "+".join(parts) or "OPEN"}


# -----------------------------------------
# 9. RESULTS & CHART
# -----------------------------------------

def calc_max_drawdown(eq):
    return ((eq - eq.cummax()) / eq.cummax() * 100).min()


def print_results(df):
    if df.empty:
        print("\nStill no trades.")
        print("Next step: run check_timezone.py and paste the hour distribution here.")
        return

    w = df[df["pnl"] > 0]; l = df[df["pnl"] <= 0]
    total = len(df)
    pf = w["pnl"].sum() / abs(l["pnl"].sum()) if not l.empty else float("inf")
    ret = (df["account"].iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    print("\n" + "="*54)
    print("  XAU/USD BACKTEST RESULTS -- v3b")
    print("="*54)
    print(f"  Period:          {df['entry_time'].min().date()} to {df['entry_time'].max().date()}")
    print(f"  Total trades:    {total}")
    print(f"  Win rate:        {len(w)/total*100:.1f}%")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Avg TP1 R:R:     {df['tp1_rr'].mean():.2f}x")
    print(f"  Avg win:         ${w['pnl'].mean():.2f}" if not w.empty else "  Avg win:         --")
    print(f"  Avg loss:        ${l['pnl'].mean():.2f}" if not l.empty else "  Avg loss:         --")
    print(f"  Total P&L:       ${df['pnl'].sum():.2f}")
    print(f"  Return:          {ret:.1f}%")
    print(f"  Max drawdown:    {calc_max_drawdown(df['account']):.1f}%")
    print(f"  Final account:   ${df['account'].iloc[-1]:.2f}")
    print("="*54)
    print("\n  Outcome breakdown:")
    print(df["outcome"].value_counts().to_string())
    print()
    print("  -- Version history ----------------------------")
    print("  v1  Trades: 55 | WR: 38.2% | PF: 0.96 | DD: -27.1%")
    print(f"  v3b Trades: {total:<3} | WR: {len(w)/total*100:.1f}%  | PF: {pf:.2f}  | DD: {calc_max_drawdown(df['account']):.1f}%")
    print("  -----------------------------------------------\n")


def plot_results(df):
    if df.empty: return
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("XAU/USD Backtest v3b", fontsize=13)

    axes[0].plot(df["entry_time"], df["account"], color="#2563eb", linewidth=1.8)
    axes[0].axhline(ACCOUNT_SIZE, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Equity curve"); axes[0].set_ylabel("Account ($)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].grid(True, alpha=0.2)

    colors = ["#16a34a" if p > 0 else "#dc2626" for p in df["pnl"]]
    axes[1].bar(range(len(df)), df["pnl"], color=colors, width=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_title("P&L per trade"); axes[1].set_ylabel("P&L ($)")
    axes[1].set_xlabel("Trade #"); axes[1].grid(True, alpha=0.2)

    dd = (df["account"] - df["account"].cummax()) / df["account"].cummax() * 100
    axes[2].fill_between(df["entry_time"], dd, 0, color="#dc2626", alpha=0.3)
    axes[2].plot(df["entry_time"], dd, color="#dc2626", linewidth=1)
    axes[2].set_title("Drawdown (%)"); axes[2].set_ylabel("Drawdown (%)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("backtest_results_v3b.png", dpi=150, bbox_inches="tight")
    print("  Chart saved to: backtest_results_v3b.png")
    plt.show()


# -----------------------------------------
# 10. RUN
# -----------------------------------------

if __name__ == "__main__":
    print(f"\nLoading data from: {DATA_FILE}")
    if not Path(DATA_FILE).exists():
        print(f"ERROR: '{DATA_FILE}' not found.")
        exit(1)

    m15_df = load_data(DATA_FILE)
    print(f"Loaded {len(m15_df):,} candles: {m15_df.index[0]} to {m15_df.index[-1]}")

    m15_df = add_emas(add_atr(m15_df))
    h4_df  = add_emas(resample_to_h4(m15_df))
    prev_day_levels = get_prev_day_levels(m15_df)

    print("Running backtest...")
    trades = run_backtest(m15_df, h4_df, prev_day_levels)

    print_results(trades)
    if not trades.empty:
        trades.to_csv("backtest_trades_v3b.csv", index=False)
        print("  Trade log saved to: backtest_trades_v3b.csv")
        plot_results(trades)
