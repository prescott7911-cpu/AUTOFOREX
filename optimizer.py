"""
XAU/USD Strategy Optimizer
Grid-searches parameter combinations and ranks by profit factor.

WARNING: Optimization on small datasets (< 6 months) will overfit.
Results should be validated on out-of-sample data before live trading.
Minimum 30 trades required before a result is considered meaningful.
"""

import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_FILE    = "XAUUSD_M15.csv"
ACCOUNT_SIZE = 10000
MIN_TRADES   = 10   # ignore results with fewer trades (likely overfit)

PIP = 0.1
H4  = "4h"

# ── PARAMETER GRID ───────────────────────────────────────────────────────────
# Add/remove values to widen or narrow the search.
# WARNING: total combinations = product of all list lengths.
# Current default: ~1,000–2,000 combinations (takes 1–3 minutes).

PARAM_GRID = {
    "fast_ema":         [10, 20],
    "slow_ema":         [50, 100],
    "risk_pct":         [0.01, 0.02],
    "pullback_tol":     [0.001, 0.002, 0.004],
    "min_rr":           [1.0, 1.5, 2.0],
    "h4_prox_pips":     [50, 100, 200],       # 200 = effectively off
    "atr_filter":       [True, False],
    "session_filter":   [True, False],
    "breakout_filter":  [True, False],
    "sweep_pips":       [2, 5],
}

SESSION_WINDOWS = [(7, 10), (13, 16)]   # London + NY open (GMT)
ATR_PERIOD    = 14
ATR_MA_PERIOD = 20


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "time"})
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close"]]


def add_indicators(df, fast, slow):
    df = df.copy()
    df[f"ema{fast}"] = df["close"].ewm(span=fast, adjust=False).mean()
    df[f"ema{slow}"] = df["close"].ewm(span=slow, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]    = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    return df


def resample_h4(df, fast, slow):
    h4 = df.resample(H4).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h4[f"ema{fast}"] = h4["close"].ewm(span=fast, adjust=False).mean()
    h4[f"ema{slow}"] = h4["close"].ewm(span=slow, adjust=False).mean()
    return h4


def get_prev_day_levels(df):
    daily = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"]  = daily["low"].shift(1)
    return daily[["prev_high", "prev_low"]]


# ── SIGNAL & FILTER HELPERS ──────────────────────────────────────────────────

def h4_trend(h4_df, ts, fast, slow):
    before = h4_df[h4_df.index <= ts]
    if before.empty:
        return None
    last = before.iloc[-1]
    if last[f"ema{fast}"] > last[f"ema{slow}"]:
        return "bull"
    if last[f"ema{fast}"] < last[f"ema{slow}"]:
        return "bear"
    return None


def bullish_engulf(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc < po and cc > co and cc > po and co < pc


def bearish_engulf(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc > po and cc < co and cc < po and co > pc


def pullback_ok(df, i, trend, tol, fast):
    if i < 3:
        return False
    for lb in [2, 1]:
        p = df["close"].iloc[i - lb]
        e = df[f"ema{fast}"].iloc[i - lb]
        if abs(p - e) / e > tol:
            return False
    return True


def tp_levels(entry, direction, ph, pl, sl_pips, min_rr, sweep_pips):
    sweep    = sweep_pips * PIP
    min_dist = sl_pips * PIP * min_rr
    if direction == "buy":
        lvls = sorted({l + sweep for l in [ph, pl] if l > entry})
        if not lvls or (lvls[0] - entry) < min_dist:
            return None
        while len(lvls) < 3:
            lvls.append(lvls[-1] + (lvls[0] - entry) * 0.5)
        return lvls[:3]
    else:
        lvls = sorted({l - sweep for l in [pl, ph] if l < entry}, reverse=True)
        if not lvls or (entry - lvls[0]) < min_dist:
            return None
        while len(lvls) < 3:
            lvls.append(lvls[-1] - (entry - lvls[0]) * 0.5)
        return lvls[:3]


# ── TRADE SIMULATION ─────────────────────────────────────────────────────────

def simulate(df, start_i, direction, entry, sl, tp1, tp2, tp3, units):
    rem = dict(units)
    pnl, sl_moved, cur_sl = 0.0, False, sl
    parts = []

    for i in range(start_i, len(df)):
        h, l, t = df["high"].iloc[i], df["low"].iloc[i], df.index[i]

        if direction == "buy":
            if l <= cur_sl:
                pnl -= sum(rem.values()) * (entry - cur_sl)
                parts.append("SL"); rem = {}; break
            if "tp1" in rem and h >= tp1:
                pnl += rem.pop("tp1") * (tp1 - entry); parts.append("TP1")
                if not sl_moved: cur_sl = entry; sl_moved = True
            if "tp2" in rem and h >= tp2:
                pnl += rem.pop("tp2") * (tp2 - entry); parts.append("TP2")
            if "tp3" in rem and h >= tp3:
                pnl += rem.pop("tp3") * (tp3 - entry); parts.append("TP3")
        else:
            if h >= cur_sl:
                pnl -= sum(rem.values()) * (cur_sl - entry)
                parts.append("SL"); rem = {}; break
            if "tp1" in rem and l <= tp1:
                pnl += rem.pop("tp1") * (entry - tp1); parts.append("TP1")
                if not sl_moved: cur_sl = entry; sl_moved = True
            if "tp2" in rem and l <= tp2:
                pnl += rem.pop("tp2") * (entry - tp2); parts.append("TP2")
            if "tp3" in rem and l <= tp3:
                pnl += rem.pop("tp3") * (entry - tp3); parts.append("TP3")

        if not rem:
            return pnl
        if i - start_i > 5 * 96:
            c = df["close"].iloc[i]
            pnl += sum(rem.values()) * ((c - entry) if direction == "buy" else (entry - c))
            return pnl

    return pnl


# ── SINGLE BACKTEST RUN ──────────────────────────────────────────────────────

def run_once(m15, h4, prev_levels, p):
    fast       = p["fast_ema"]
    slow       = p["slow_ema"]
    risk       = p["risk_pct"]
    tol        = p["pullback_tol"]
    min_rr     = p["min_rr"]
    h4_prox    = p["h4_prox_pips"]
    use_atr    = p["atr_filter"]
    use_sess   = p["session_filter"]
    use_bo     = p["breakout_filter"]
    sweep_pips = p["sweep_pips"]

    account = ACCOUNT_SIZE
    wins, losses, pnls = 0, 0, []
    min_i = max(slow, ATR_MA_PERIOD) + 3

    for i in range(min_i, len(m15)):
        ts   = m15.index[i]
        date = ts.date()

        if use_sess and not any(s <= ts.hour < e for s, e in SESSION_WINDOWS):
            continue

        trend = h4_trend(h4, ts, fast, slow)
        if trend is None:
            continue

        if use_atr and (pd.isna(m15["atr_ma"].iloc[i]) or m15["atr"].iloc[i] <= m15["atr_ma"].iloc[i]):
            continue

        if not pullback_ok(m15, i, trend, tol, fast):
            continue

        signal = None
        if trend == "bull" and bullish_engulf(m15, i):
            signal = "buy"
        elif trend == "bear" and bearish_engulf(m15, i):
            signal = "sell"
        if signal is None:
            continue

        close = m15["close"].iloc[i]
        e_fast = m15[f"ema{fast}"].iloc[i]
        e_slow = m15[f"ema{slow}"].iloc[i]
        if signal == "buy"  and not (close > e_fast > e_slow): continue
        if signal == "sell" and not (close < e_fast < e_slow): continue

        h4_before = h4[h4.index <= ts]
        if not h4_before.empty:
            h4_ema = h4_before[f"ema{fast}"].iloc[-1]
            if abs(close - h4_ema) / PIP > h4_prox:
                continue

        prev = prev_levels[prev_levels.index.date == date]
        if prev.empty: continue
        ph = prev["prev_high"].iloc[0]
        pl = prev["prev_low"].iloc[0]
        if pd.isna(ph) or pd.isna(pl): continue

        if use_bo:
            if signal == "buy"  and close < ph: continue
            if signal == "sell" and close > pl: continue

        entry = close
        sl    = m15["low"].iloc[i-1]  if signal == "buy"  else m15["high"].iloc[i-1]
        sl_pips = abs(entry - sl) / PIP
        if sl_pips <= 0: continue

        tps = tp_levels(entry, signal, ph, pl, sl_pips, min_rr, sweep_pips)
        if tps is None: continue

        tp1, tp2, tp3 = tps
        risk_amt = account * risk
        units_total = risk_amt / (sl_pips * PIP)
        units = {"tp1": units_total * 0.5, "tp2": units_total * 0.3, "tp3": units_total * 0.2}

        pnl = simulate(m15, i + 1, signal, entry, sl, tp1, tp2, tp3, units)
        account += pnl
        pnls.append(pnl)
        if pnl > 0: wins += 1
        else: losses += 1

    total = wins + losses
    if total < MIN_TRADES:
        return None

    win_rate = wins / total
    gross_win  = sum(x for x in pnls if x > 0)
    gross_loss = abs(sum(x for x in pnls if x <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    eq = pd.Series([ACCOUNT_SIZE] + [ACCOUNT_SIZE + sum(pnls[:k+1]) for k in range(len(pnls))])
    max_dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    ret = (account - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    return {
        "trades":   total,
        "win_rate": round(win_rate * 100, 1),
        "pf":       round(pf, 2),
        "return":   round(ret, 2),
        "max_dd":   round(max_dd, 1),
        "score":    round(pf * win_rate * (1 + ret / 100), 4),  # composite score
        **p,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading {DATA_FILE}...")
    if not Path(DATA_FILE).exists():
        print(f"ERROR: {DATA_FILE} not found. Run download_data.py first.")
        exit(1)

    raw = load_data(DATA_FILE)
    print(f"Loaded {len(raw):,} candles.")

    # Pre-compute indicator sets per unique EMA combo to avoid recomputing
    ema_combos = set(
        (p["fast_ema"], p["slow_ema"])
        for p in [dict(zip(PARAM_GRID.keys(), v)) for v in itertools.product(*PARAM_GRID.values())]
    )
    print(f"Pre-computing indicators for {len(ema_combos)} EMA combos...")
    indicator_cache = {}
    h4_cache = {}
    for fast, slow in ema_combos:
        indicator_cache[(fast, slow)] = add_indicators(raw, fast, slow)
        h4_cache[(fast, slow)]        = resample_h4(raw, fast, slow)

    prev_levels = get_prev_day_levels(raw)

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total_combos = len(combos)
    print(f"Scanning {total_combos:,} parameter combinations...")
    print(f"Minimum trades required per result: {MIN_TRADES}\n")

    results = []
    for idx, values in enumerate(combos):
        p = dict(zip(keys, values))
        if p["fast_ema"] >= p["slow_ema"]:
            continue

        m15 = indicator_cache[(p["fast_ema"], p["slow_ema"])]
        h4  = h4_cache[(p["fast_ema"], p["slow_ema"])]

        res = run_once(m15, h4, prev_levels, p)
        if res:
            results.append(res)

        if (idx + 1) % 100 == 0 or idx + 1 == total_combos:
            valid = len(results)
            print(f"  {idx+1:>5}/{total_combos}  valid results so far: {valid}", end="\r")

    print(f"\n\nDone. {len(results)} combinations met the {MIN_TRADES}-trade minimum.\n")

    if not results:
        print("No valid results found.")
        print("Try reducing MIN_TRADES or adding more historical data.")
        exit(0)

    df = pd.DataFrame(results).sort_values("score", ascending=False)

    print("=" * 80)
    print("  TOP 10 STRATEGY COMBINATIONS  (ranked by composite score)")
    print("  Score = profit_factor x win_rate x (1 + return)")
    print("=" * 80)
    cols = ["score", "trades", "win_rate", "pf", "return", "max_dd",
            "fast_ema", "slow_ema", "risk_pct", "pullback_tol",
            "min_rr", "h4_prox_pips", "atr_filter", "session_filter",
            "breakout_filter", "sweep_pips"]
    print(df[cols].head(10).to_string(index=False))

    print("\n")
    print("  TOP 10 BY PROFIT FACTOR")
    print("-" * 80)
    print(df[cols].sort_values("pf", ascending=False).head(10).to_string(index=False))

    print("\n")
    print("  TOP 10 BY RETURN %")
    print("-" * 80)
    print(df[cols].sort_values("return", ascending=False).head(10).to_string(index=False))

    out_file = "optimizer_results.csv"
    df[cols].to_csv(out_file, index=False)
    print(f"\n  Full results saved to: {out_file}")

    best = df.iloc[0]
    print("\n" + "=" * 80)
    print("  BEST COMBINATION (copy these into backtest_v3b.py)")
    print("=" * 80)
    print(f"  FAST_EMA              = {int(best['fast_ema'])}")
    print(f"  SLOW_EMA              = {int(best['slow_ema'])}")
    print(f"  RISK_PCT              = {best['risk_pct']}")
    print(f"  PULLBACK_TOL          = {best['pullback_tol']}")
    print(f"  MIN_RR                = {best['min_rr']}")
    print(f"  H4_EMA_PROXIMITY_PIPS = {int(best['h4_prox_pips'])}")
    print(f"  ATR filter            = {best['atr_filter']}")
    print(f"  Session filter        = {best['session_filter']}")
    print(f"  Breakout filter       = {best['breakout_filter']}")
    print(f"  SWEEP_PIPS            = {int(best['sweep_pips'])}")
    print(f"\n  Expected (in-sample): {best['trades']} trades | "
          f"WR {best['win_rate']}% | PF {best['pf']} | "
          f"Return {best['return']}% | DD {best['max_dd']}%")
    print("\n  !! WARNING: These results are IN-SAMPLE only.")
    print("  !! Get 6-12 months of data and validate on unseen data before trading live.")
    print("=" * 80)
