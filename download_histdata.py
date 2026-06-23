"""
Convert locally downloaded histdata.com zip files to XAUUSD_M15.csv.

STEP 1 — Download the data manually:
  1. Go to: https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/XAUUSD
  2. Select a year (e.g. 2024), download each month's zip file
  3. Repeat for 2025 and 2026
  4. Put all zip files into: C:\\Users\\presc\\AUTOFOREX\\histdata\\

STEP 2 — Run this script:
  python download_histdata.py

Output: XAUUSD_M15.csv (~35,000 candles, ready for optimizer.py)
"""

import zipfile
import pandas as pd
from pathlib import Path
import sys

HISTDATA_DIR = Path("histdata")
OUTPUT_FILE  = "XAUUSD_M15.csv"
RESAMPLE_TF  = "15min"


def parse_zip(zip_path):
    """Extract and parse one histdata zip file. Returns M1 DataFrame or None."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Find the data file inside the zip
            data_file = next(
                (n for n in zf.namelist() if n.upper().endswith(".CSV") or n.upper().endswith(".DAT")),
                None
            )
            if not data_file:
                print(f"    No CSV/DAT found in {zip_path.name}")
                return None

            raw = zf.read(data_file).decode("utf-8", errors="ignore")

        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            return None

        rows = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 6:
                rows.append(parts[:6])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close"])

        # histdata format: YYYYMMDD, HHMMSS
        df["datetime"] = pd.to_datetime(
            df["date"].str.strip() + " " + df["time"].str.strip(),
            format="%Y%m%d %H%M%S",
            errors="coerce"
        )
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close"]].dropna()
        return df

    except Exception as e:
        print(f"    Error reading {zip_path.name}: {e}")
        return None


def resample_to_m15(m1_df):
    return m1_df.resample(RESAMPLE_TF).agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }).dropna()


if __name__ == "__main__":
    if not HISTDATA_DIR.exists():
        print(f"\nERROR: Folder '{HISTDATA_DIR}' not found.")
        print("Create it and place your histdata.com zip files inside:")
        print(f"  mkdir {HISTDATA_DIR}")
        print("\nThen download zips from:")
        print("  https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/XAUUSD")
        sys.exit(1)

    zips = sorted(HISTDATA_DIR.glob("*.zip"))
    if not zips:
        print(f"\nNo zip files found in '{HISTDATA_DIR}/'.")
        print("Download monthly zip files from histdata.com and place them there.")
        sys.exit(1)

    print(f"\nFound {len(zips)} zip file(s) in '{HISTDATA_DIR}/':")
    for z in zips:
        print(f"  {z.name}")

    print(f"\nParsing M1 data...")
    frames = []
    for z in zips:
        print(f"  {z.name}...", end=" ", flush=True)
        df = parse_zip(z)
        if df is not None and not df.empty:
            print(f"{len(df):,} bars")
            frames.append(df)
        else:
            print("skipped")

    if not frames:
        print("\nNo data could be parsed. Check your zip files are from histdata.com (XAUUSD, M1, ASCII).")
        sys.exit(1)

    print(f"\nCombining {len(frames)} month(s) of M1 data...")
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    print(f"Total M1 bars: {len(m1):,}")
    print(f"Range: {m1.index[0]} to {m1.index[-1]}")

    print(f"Resampling to M15...")
    m15 = resample_to_m15(m1)
    m15 = m15.reset_index().rename(columns={"datetime": "datetime"})
    m15.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(m15):,} M15 candles to '{OUTPUT_FILE}'")
    print(f"Ready to run: python optimizer.py")
