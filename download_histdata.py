"""
Download 1-2 years of XAU/USD M1 data from histdata.com and resample to M15.
Saves to XAUUSD_M15.csv, ready for backtest.py and optimizer.py.

Usage:
    python download_histdata.py

No account needed. Downloads ~12 monthly zip files (~50MB total).
"""

import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import time

OUTPUT_FILE  = "XAUUSD_M15.csv"
PAIR         = "XAUUSD"
YEARS        = 2          # how many years back to fetch (max ~5 on histdata)
RESAMPLE_TF  = "15min"
SLEEP_SEC    = 1.5        # polite delay between requests

HISTDATA_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}/{month}"
HISTDATA_POST = "https://www.histdata.com/get.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://www.histdata.com/",
}


def fetch_month(session, pair, year, month):
    """Download one month of M1 data. Returns DataFrame or None."""
    ref_url = HISTDATA_URL.format(pair=pair.lower(), year=year, month=month)

    try:
        # POST to get the zip
        data = {
            "tk":   _get_token(session, ref_url),
            "date": str(year),
            "datemonth": f"{month:02d}",
            "platform": "ASCII",
            "timeframe": "M1",
            "fxpair": pair,
        }
        resp = session.post(HISTDATA_POST, data=data, headers={**HEADERS, "Referer": ref_url}, timeout=30)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return None

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next((n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".DAT")), None)
        if not csv_name:
            return None

        raw = zf.read(csv_name).decode("utf-8", errors="ignore")
        lines = [l for l in raw.strip().splitlines() if l.strip()]

        rows = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 6:
                rows.append(parts[:6])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close"])
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close"]].dropna()
        return df

    except Exception as e:
        print(f"    Error: {e}")
        return None


def _get_token(session, ref_url):
    """Extract the CSRF-like token from the histdata page."""
    try:
        r = session.get(ref_url, headers=HEADERS, timeout=15)
        for line in r.text.splitlines():
            if "id=\"tk\"" in line:
                start = line.find('value="') + 7
                end   = line.find('"', start)
                return line[start:end]
    except Exception:
        pass
    return ""


def resample_to_m15(m1_df):
    return m1_df.resample(RESAMPLE_TF).agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna()


if __name__ == "__main__":
    now   = date.today()
    pairs = []
    for y in range(now.year, now.year - YEARS - 1, -1):
        for m in range(12, 0, -1):
            d = date(y, m, 1)
            if d > now:
                continue
            if date(y, m, 1) < date(now.year - YEARS, now.month, 1):
                break
            pairs.append((y, m))
    pairs = sorted(pairs)

    print(f"\nDownloading {PAIR} M1 data from histdata.com")
    print(f"Period: {pairs[0][0]}-{pairs[0][1]:02d} to {pairs[-1][0]}-{pairs[-1][1]:02d}")
    print(f"Months to fetch: {len(pairs)}\n")

    session = requests.Session()
    frames  = []

    for year, month in pairs:
        label = f"{year}-{month:02d}"
        print(f"  Fetching {label}...", end=" ", flush=True)
        df = fetch_month(session, PAIR, year, month)
        if df is not None and not df.empty:
            print(f"{len(df):,} M1 bars")
            frames.append(df)
        else:
            print("no data / skipped")
        time.sleep(SLEEP_SEC)

    if not frames:
        print("\nNo data downloaded. Possible causes:")
        print("  - histdata.com changed their page structure")
        print("  - Network blocked the requests")
        print("  - Try downloading manually from https://www.histdata.com")
        print("    Select: Forex > XAUUSD > 1 Minute OHLC > year, then unzip and")
        print("    save the .csv files in a 'histdata/' folder, then run:")
        print("    python download_histdata.py --local histdata/")
        exit(1)

    print(f"\nCombining {len(frames)} months of M1 data...")
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    print(f"Total M1 bars: {len(m1):,}")

    print(f"Resampling to M15...")
    m15 = resample_to_m15(m1)
    m15 = m15.reset_index().rename(columns={"datetime": "datetime"})
    m15.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(m15):,} M15 candles to {OUTPUT_FILE}")
    print(f"Range: {m15['datetime'].iloc[0]} to {m15['datetime'].iloc[-1]}")
    print(f"\nNow run: python optimizer.py")
