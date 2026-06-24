"""
XAU/USD Live Trader — MT5
Uses the same EMA 10/50 crossover + pullback + engulfing signal as the backtest.
Runs every 15 minutes, checks for a signal on the just-closed M15 candle,
and places a market order with SL and 3 TP levels.

SETUP:
  1. Open MetaTrader 5 and log into your broker account
  2. Make sure XAUUSD is visible in Market Watch
  3. Run: python live_trader.py

SAFETY:
  - Set LIVE_TRADE = False to run in signal-only mode (no orders placed)
  - Start with a demo account to verify signals match your chart
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone

# ── CONFIGURATION ────────────────────────────────────────────────────────────

SYMBOL       = "XAUUSD"
LIVE_TRADE   = False          # !! SET TO True ONLY when ready to trade live
ACCOUNT_SIZE = 10000          # used for position sizing (update to real balance)
RISK_PCT     = 0.005          # 0.5% risk per trade (conservative)

FAST_EMA   = 10
SLOW_EMA   = 50
PULLBACK_TOL = 0.001
MIN_RR     = 1.0
SWEEP_PIPS = 5
PIP        = 0.1

TP1_PCT = 0.50
TP2_PCT = 0.30
TP3_PCT = 0.20

MAGIC      = 202501           # unique ID for this bot's orders
COMMENT    = "AUTOFOREX_v3b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("live_trader.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ── MT5 CONNECTION ───────────────────────────────────────────────────────────

def connect():
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        log.error("Not logged in. Open MT5 and log into your account first.")
        mt5.shutdown()
        return False
    log.info(f"Connected: {info.name} | Balance: {info.balance} {info.currency} | Server: {info.server}")
    return True


def disconnect():
    mt5.shutdown()
    log.info("MT5 disconnected.")


# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def get_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    return df[["open", "high", "low", "close"]]


def add_emas(df):
    df = df.copy()
    df["ef"] = df["close"].ewm(span=FAST_EMA, adjust=False).mean()
    df["es"] = df["close"].ewm(span=SLOW_EMA, adjust=False).mean()
    return df


# ── SIGNAL LOGIC (mirrors backtest exactly) ──────────────────────────────────

def get_h4_trend(h4_df):
    last = h4_df.iloc[-2]   # use confirmed closed candle
    if last["ef"] > last["es"]:
        return "bull"
    if last["ef"] < last["es"]:
        return "bear"
    return None


def pullback_ok(m15_df, i):
    trend = get_h4_trend_from_m15(m15_df)
    if trend is None:
        return False
    for lb in [2, 1]:
        price = m15_df["close"].iloc[i - lb]
        ema   = m15_df["ef"].iloc[i - lb]
        if abs(price - ema) / ema > PULLBACK_TOL:
            return False
    return True


def get_h4_trend_from_m15(m15_df):
    h4 = m15_df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","ef":"last","es":"last"}).dropna()
    if len(h4) < 2:
        return None
    last = h4.iloc[-2]
    if last["ef"] > last["es"]: return "bull"
    if last["ef"] < last["es"]: return "bear"
    return None


def is_bullish_engulfing(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc < po and cc > co and cc > po and co < pc


def is_bearish_engulfing(df, i):
    po, pc = df["open"].iloc[i-1], df["close"].iloc[i-1]
    co, cc = df["open"].iloc[i],   df["close"].iloc[i]
    return pc > po and cc < co and cc < po and co > pc


def get_prev_day_levels(m15_df):
    daily = m15_df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    if len(daily) < 2:
        return None, None
    return daily["high"].iloc[-2], daily["low"].iloc[-2]


def select_tps(entry, direction, ph, pl, sl_pips):
    sweep    = SWEEP_PIPS * PIP
    min_dist = sl_pips * PIP * MIN_RR
    if direction == "buy":
        lvls = sorted({l + sweep for l in [ph, pl] if l > entry})
        if not lvls or (lvls[0] - entry) < min_dist: return None
        while len(lvls) < 3: lvls.append(lvls[-1] + (lvls[0] - entry) * 0.5)
        return lvls[:3]
    else:
        lvls = sorted({l - sweep for l in [pl, ph] if l < entry}, reverse=True)
        if not lvls or (entry - lvls[0]) < min_dist: return None
        while len(lvls) < 3: lvls.append(lvls[-1] - (entry - lvls[0]) * 0.5)
        return lvls[:3]


def check_signal():
    """Fetch latest M15 data and check for a trade signal on the last closed candle."""
    df = get_rates(SYMBOL, mt5.TIMEFRAME_M15, SLOW_EMA + 20)
    if df is None:
        log.warning("Failed to fetch M15 data.")
        return None

    df = add_emas(df)
    i = -2   # last CLOSED candle (index -1 is the forming candle)

    trend = get_h4_trend_from_m15(df)
    if trend is None:
        return None

    if not pullback_ok(df, i):
        return None

    signal = None
    if trend == "bull" and is_bullish_engulfing(df, i):
        signal = "buy"
    elif trend == "bear" and is_bearish_engulfing(df, i):
        signal = "sell"

    if signal is None:
        return None

    close = df["close"].iloc[i]
    ef    = df["ef"].iloc[i]
    es    = df["es"].iloc[i]
    if signal == "buy"  and not (close > ef > es): return None
    if signal == "sell" and not (close < ef < es): return None

    ph, pl = get_prev_day_levels(df)
    if ph is None: return None

    sl = df["low"].iloc[i-1]  if signal == "buy"  else df["high"].iloc[i-1]
    sl_pips = abs(close - sl) / PIP
    if sl_pips <= 0: return None

    tps = select_tps(close, signal, ph, pl, sl_pips)
    if tps is None: return None

    return {
        "signal":   signal,
        "entry":    close,
        "sl":       sl,
        "tp1":      tps[0],
        "tp2":      tps[1],
        "tp3":      tps[2],
        "sl_pips":  round(sl_pips, 1),
        "time":     df.index[i],
    }


# ── ORDER EXECUTION ──────────────────────────────────────────────────────────

def get_lot_size(sl_pips):
    """Convert risk % to MT5 lot size for XAUUSD."""
    info = mt5.account_info()
    balance = info.balance if info else ACCOUNT_SIZE
    risk_amount = balance * RISK_PCT

    tick = mt5.symbol_info(SYMBOL)
    if tick is None:
        log.error(f"Symbol {SYMBOL} not found in MT5.")
        return 0.01

    # XAUUSD: 1 lot = 100 oz, pip value ~$1 per 0.01 lot per pip
    # pip_value per lot = tick.trade_tick_value / tick.trade_tick_size * PIP
    pip_value_per_lot = (tick.trade_tick_value / tick.trade_tick_size) * PIP
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    lot = round(max(tick.volume_min, min(lot, tick.volume_max)), 2)
    return lot


def place_order(sig, lot_fraction=1.0, tp=None, comment_suffix=""):
    """Place a single market order with SL and one TP."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error("Failed to get tick data.")
        return None

    price  = tick.ask if sig["signal"] == "buy" else tick.bid
    action = mt5.ORDER_TYPE_BUY if sig["signal"] == "buy" else mt5.ORDER_TYPE_SELL
    sl_pips = sig["sl_pips"]
    lot    = round(get_lot_size(sl_pips) * lot_fraction, 2)
    if lot <= 0:
        return None

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    SYMBOL,
        "volume":    lot,
        "type":      action,
        "price":     price,
        "sl":        sig["sl"],
        "tp":        tp,
        "magic":     MAGIC,
        "comment":   f"{COMMENT}{comment_suffix}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed [{comment_suffix}]: retcode={result.retcode} | {result.comment}")
        return None

    log.info(f"Order placed [{comment_suffix}]: {sig['signal'].upper()} {lot} lots @ {price} | SL={sig['sl']} TP={tp}")
    return result


def execute_signal(sig):
    """Split into 3 orders: 50% to TP1, 30% to TP2, 20% to TP3."""
    log.info(f"SIGNAL: {sig['signal'].upper()} | Entry={sig['entry']} SL={sig['sl']} "
             f"TP1={sig['tp1']:.2f} TP2={sig['tp2']:.2f} TP3={sig['tp3']:.2f} | {sig['time']}")

    if not LIVE_TRADE:
        log.info("LIVE_TRADE=False — signal logged but no order placed.")
        return

    place_order(sig, lot_fraction=TP1_PCT, tp=sig["tp1"], comment_suffix="_TP1")
    place_order(sig, lot_fraction=TP2_PCT, tp=sig["tp2"], comment_suffix="_TP2")
    place_order(sig, lot_fraction=TP3_PCT, tp=sig["tp3"], comment_suffix="_TP3")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def seconds_to_next_candle():
    """Seconds until the next M15 candle closes (plus 5s buffer)."""
    now = datetime.now(timezone.utc)
    elapsed = (now.minute % 15) * 60 + now.second
    return (15 * 60 - elapsed) + 5


def run():
    log.info("=" * 60)
    log.info(f"AUTOFOREX Live Trader starting | LIVE_TRADE={LIVE_TRADE}")
    log.info(f"Symbol: {SYMBOL} | Risk: {RISK_PCT*100}% | EMA: {FAST_EMA}/{SLOW_EMA}")
    if not LIVE_TRADE:
        log.info("Running in SIGNAL-ONLY mode. Set LIVE_TRADE=True to place orders.")
    log.info("=" * 60)

    if not connect():
        return

    try:
        while True:
            wait = seconds_to_next_candle()
            log.info(f"Next candle in {wait}s — waiting...")
            time.sleep(wait)

            sig = check_signal()
            if sig:
                execute_signal(sig)
            else:
                log.info("No signal on this candle.")

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        disconnect()


if __name__ == "__main__":
    run()
