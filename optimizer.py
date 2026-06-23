"""
XAU/USD Strategy Optimizer
Grid-searches parameter combinations and ranks by composite score.

WARNING: Optimization on small datasets (< 6 months) will overfit.
Validate results on out-of-sample data before live trading.
Minimum 10 trades required before a result is considered meaningful.

Performance: all H4 trend and EMA lookups are pre-computed as arrays
so the inner loop does only array indexing — no DataFrame scanning.
~1,400 combos on 4,500 candles should complete in under 60 seconds.
"""

import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_FILE    = "XAUUSD_M15.csv"
ACCOUNT_SIZE = 10000
MIN_TRADES   = 10

PIP = 0.1
H4  = "4h"

PARAM_GRID = {
    "fast_ema":        [10, 20],
    "slow_ema":        [50, 100],
    "risk_pct":        [0.01, 0.02],
    "pullback_tol":    [0.001, 0.002, 0.004],
    "min_rr":          [1.0, 1.5, 2.0],
    "h4_prox_pips":    [50, 100, 999],    # 999 = effectively off
    "atr_filter":      [True, False],
    "session_filter":  [True, False],
    "breakout_filter": [True, False],
    "sweep_pips":      [2, 5],
}

SESSION_WINDOWS = [(7, 10), (13, 16)]
ATR_PERIOD    = 14
ATR_MA_PERIOD = 20


# ── DATA & INDICATORS ────────────────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "time"})
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close"]].copy()


def build_cache(df, fast, slow):
    """
    Pre-compute everything needed for the inner loop as numpy arrays.
    H4 trend and H4 EMA are forward-filled to M15 index via merge_asof.
    """
    d = df.copy()

    # M15 EMAs
    d[f"ef"] = d["close"].ewm(span=fast, adjust=False).mean()
    d[f"es"] = d["close"].ewm(span=slow, adjust=False).mean()

    # ATR
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-pc).abs(), (d["low"]-pc).abs()], axis=1).max(axis=1)
    d["atr"]    = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    d["atr_ma"] = d["atr"].rolling(ATR_MA_PERIOD).mean()

    # H4 frame
    h4 = d.resample(H4).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    h4["ef"] = h4["close"].ewm(span=fast, adjust=False).mean()
    h4["es"] = h4["close"].ewm(span=slow, adjust=False).mean()
    h4["trend"] = np.where(h4["ef"] > h4["es"], 1,
                  np.where(h4["ef"] < h4["es"], -1, 0))
    h4_lookup = h4[["trend", "ef"]].reset_index().rename(columns={"time": "h4_time", "ef": "h4_ef"})

    # merge_asof: for each M15 bar, get the most recent H4 bar
    m15_reset = d.reset_index()
    merged = pd.merge_asof(m15_reset, h4_lookup, left_on="time", right_on="h4_time")
    merged = merged.set_index("time")

    d["h4_trend"] = merged["trend"].values
    d["h4_ema"]   = merged["h4_ef"].values

    # Previous day high/low — forward-fill to M15
    daily = d.resample("1D").agg({"high":"max","low":"min"}).dropna()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"]  = daily["low"].shift(1)
    dl = daily[["prev_high","prev_low"]].reset_index().rename(columns={"time":"day"})
    m15r = d.reset_index()
    m15r["day"] = m15r["time"].dt.normalize()
    m15r = m15r.merge(dl, on="day", how="left")
    m15r = m15r.set_index("time")
    d["prev_high"] = m15r["prev_high"].values
    d["prev_low"]  = m15r["prev_low"].values

    # Session mask
    d["in_session"] = d.index.hour.map(
        lambda h: any(s <= h < e for s, e in SESSION_WINDOWS)
    )

    # Engulfing signals (vectorized)
    po = d["open"].shift(1).values
    pc = d["close"].shift(1).values
    co = d["open"].values
    cc = d["close"].values
    d["bull_eng"] = (pc < po) & (cc > co) & (cc > po) & (co < pc)
    d["bear_eng"] = (pc > po) & (cc < co) & (cc < po) & (co > pc)

    return d


# ── TRADE SIMULATION (numpy-level) ──────────────────────────────────────────

def simulate(highs, lows, closes, start_i, direction, entry, sl, tp1, tp2, tp3, units):
    rem = {"tp1": units*0.5, "tp2": units*0.3, "tp3": units*0.2}
    pnl, sl_moved, cur_sl = 0.0, False, sl
    n = len(highs)

    for i in range(start_i, n):
        h, l = highs[i], lows[i]

        if direction == 1:  # buy
            if l <= cur_sl:
                pnl -= sum(rem.values()) * (entry - cur_sl)
                return pnl
            if "tp1" in rem and h >= tp1:
                pnl += rem.pop("tp1") * (tp1 - entry)
                if not sl_moved: cur_sl = entry; sl_moved = True
            if "tp2" in rem and h >= tp2:
                pnl += rem.pop("tp2") * (tp2 - entry)
            if "tp3" in rem and h >= tp3:
                pnl += rem.pop("tp3") * (tp3 - entry)
        else:  # sell
            if h >= cur_sl:
                pnl -= sum(rem.values()) * (cur_sl - entry)
                return pnl
            if "tp1" in rem and l <= tp1:
                pnl += rem.pop("tp1") * (entry - tp1)
                if not sl_moved: cur_sl = entry; sl_moved = True
            if "tp2" in rem and l <= tp2:
                pnl += rem.pop("tp2") * (entry - tp2)
            if "tp3" in rem and l <= tp3:
                pnl += rem.pop("tp3") * (entry - tp3)

        if not rem:
            return pnl
        if i - start_i > 480:   # 5-day timeout
            c = closes[i]
            pnl += sum(rem.values()) * ((c - entry) if direction == 1 else (entry - c))
            return pnl

    return pnl


# ── SINGLE RUN ───────────────────────────────────────────────────────────────

def run_once(d, p):
    fast    = p["fast_ema"]
    risk    = p["risk_pct"]
    tol     = p["pullback_tol"]
    min_rr  = p["min_rr"]
    h4_prox = p["h4_prox_pips"]
    sweep   = p["sweep_pips"] * PIP
    use_atr  = p["atr_filter"]
    use_sess = p["session_filter"]
    use_bo   = p["breakout_filter"]

    # Extract numpy arrays once
    highs     = d["high"].values
    lows      = d["low"].values
    closes    = d["close"].values
    opens     = d["open"].values
    ef        = d["ef"].values
    es        = d["es"].values
    atr       = d["atr"].values
    atr_ma    = d["atr_ma"].values
    h4_trend  = d["h4_trend"].values
    h4_ema    = d["h4_ema"].values
    prev_high = d["prev_high"].values
    prev_low  = d["prev_low"].values
    in_sess   = d["in_session"].values
    bull_eng  = d["bull_eng"].values
    bear_eng  = d["bear_eng"].values

    account = ACCOUNT_SIZE
    pnls = []
    min_i = max(p["slow_ema"], ATR_MA_PERIOD) + 3

    for i in range(min_i, len(d)):
        if use_sess and not in_sess[i]:
            continue

        trend = h4_trend[i]
        if trend == 0:
            continue

        if use_atr and (np.isnan(atr_ma[i]) or atr[i] <= atr_ma[i]):
            continue

        # 2-candle pullback check
        if i < 3:
            continue
        pb_ok = True
        for lb in [2, 1]:
            p_price = closes[i - lb]
            p_ema   = ef[i - lb]
            if abs(p_price - p_ema) / p_ema > tol:
                pb_ok = False; break
        if not pb_ok:
            continue

        # Engulfing + trend alignment
        if trend == 1 and bull_eng[i]:
            signal = 1
        elif trend == -1 and bear_eng[i]:
            signal = -1
        else:
            continue

        close = closes[i]
        if signal == 1  and not (close > ef[i] > es[i]): continue
        if signal == -1 and not (close < ef[i] < es[i]): continue

        # H4 EMA proximity
        if abs(close - h4_ema[i]) / PIP > h4_prox:
            continue

        ph = prev_high[i]
        pl = prev_low[i]
        if np.isnan(ph) or np.isnan(pl):
            continue

        # Breakout filter
        if use_bo:
            if signal == 1  and close < ph: continue
            if signal == -1 and close > pl: continue

        entry = close
        sl    = lows[i-1]  if signal == 1 else highs[i-1]
        sl_pips = abs(entry - sl) / PIP
        if sl_pips <= 0:
            continue

        min_dist = sl_pips * PIP * min_rr

        # TP levels
        if signal == 1:
            lvls = sorted({l + sweep for l in [ph, pl] if l > entry})
            if not lvls or (lvls[0] - entry) < min_dist: continue
            while len(lvls) < 3: lvls.append(lvls[-1] + (lvls[0] - entry) * 0.5)
        else:
            lvls = sorted({l - sweep for l in [pl, ph] if l < entry}, reverse=True)
            if not lvls or (entry - lvls[0]) < min_dist: continue
            while len(lvls) < 3: lvls.append(lvls[-1] - (entry - lvls[0]) * 0.5)

        tp1, tp2, tp3 = lvls[:3]
        units_total = (account * risk) / (sl_pips * PIP)

        pnl = simulate(highs, lows, closes, i+1, signal, entry, sl, tp1, tp2, tp3, units_total)
        account += pnl
        pnls.append(pnl)

    total = len(pnls)
    if total < MIN_TRADES:
        return None

    wins = sum(1 for x in pnls if x > 0)
    gw   = sum(x for x in pnls if x > 0)
    gl   = abs(sum(x for x in pnls if x <= 0))
    pf   = gw / gl if gl > 0 else 9.99
    wr   = wins / total
    ret  = (account - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100
    eq   = np.array([ACCOUNT_SIZE + sum(pnls[:k]) for k in range(total+1)])
    peak = np.maximum.accumulate(eq)
    dd   = ((eq - peak) / peak * 100).min()

    return {
        "trades":  total,
        "win_rate": round(wr * 100, 1),
        "pf":       round(min(pf, 9.99), 2),
        "return":   round(ret, 2),
        "max_dd":   round(dd, 1),
        "score":    round(min(pf, 9.99) * wr * max(1 + ret/100, 0.01), 4),
        **p,
    }


# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nLoading {DATA_FILE}...")
    if not Path(DATA_FILE).exists():
        print(f"ERROR: {DATA_FILE} not found. Run download_data.py first.")
        exit(1)

    raw = load_data(DATA_FILE)
    print(f"Loaded {len(raw):,} candles.\n")

    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    combos = [p for p in combos if p["fast_ema"] < p["slow_ema"]]
    total  = len(combos)

    ema_pairs = sorted({(p["fast_ema"], p["slow_ema"]) for p in combos})
    print(f"Pre-computing indicators for {len(ema_pairs)} EMA pair(s)...")
    cache = {}
    for fast, slow in ema_pairs:
        cache[(fast, slow)] = build_cache(raw, fast, slow)
    print(f"Done. Scanning {total} combinations...\n")

    results = []
    for idx, p in enumerate(combos):
        d   = cache[(p["fast_ema"], p["slow_ema"])]
        res = run_once(d, p)
        if res:
            results.append(res)
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  {idx+1:>5}/{total}  valid so far: {len(results)}", end="\r")

    print(f"\n\nDone. {len(results)} combinations met the {MIN_TRADES}-trade minimum.\n")

    if not results:
        print("No valid results. Try lowering MIN_TRADES or getting more data.")
        exit(0)

    cols = ["score","trades","win_rate","pf","return","max_dd",
            "fast_ema","slow_ema","risk_pct","pullback_tol",
            "min_rr","h4_prox_pips","atr_filter","session_filter",
            "breakout_filter","sweep_pips"]

    df = pd.DataFrame(results)

    print("=" * 80)
    print("  TOP 10 BY COMPOSITE SCORE  (PF x WR x return)")
    print("=" * 80)
    print(df[cols].sort_values("score", ascending=False).head(10).to_string(index=False))

    print("\n  TOP 10 BY PROFIT FACTOR")
    print("-" * 80)
    print(df[cols].sort_values("pf", ascending=False).head(10).to_string(index=False))

    print("\n  TOP 10 BY RETURN %")
    print("-" * 80)
    print(df[cols].sort_values("return", ascending=False).head(10).to_string(index=False))

    df[cols].sort_values("score", ascending=False).to_csv("optimizer_results.csv", index=False)
    print(f"\n  Full results saved to: optimizer_results.csv")

    best = df.sort_values("score", ascending=False).iloc[0]
    print("\n" + "=" * 80)
    print("  BEST COMBINATION (paste into backtest_v3b.py)")
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
    print(f"\n  In-sample: {int(best['trades'])} trades | WR {best['win_rate']}% | "
          f"PF {best['pf']} | Return {best['return']}% | DD {best['max_dd']}%")
    print("\n  !! WARNING: IN-SAMPLE ONLY. Get 6-12 months of data before trading live.")
    print("=" * 80)
