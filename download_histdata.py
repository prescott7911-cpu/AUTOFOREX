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
import io
import pandas as pd
from pathlib import Path
import sys

HISTDATA_DIR = Path("histdata")
OUTPUT_FILE  = "XAUUSD_M15.csv"
RESAMPLE_TF  = "15min"


def parse_zip(zip_path):
    """Extract and parse one histdata zip file (annual or monthly). Returns M1 DataFrame or None."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()

            # Annual zip: contains monthly zips inside — recurse into each
            inner_zips = [n for n in names if n.upper().endswith(".ZIP")]
            if inner_zips:
                frames = []
                for inner_name in sorted(inner_zips):
                    inner_bytes = io.BytesIO(zf.read(inner_name))
                    try:
                        with zipfile.ZipFile(inner_bytes) as inner_zf:
                            df = _parse_open_zip(inner_zf, inner_name)
                            if df is not None:
                                frames.append(df)
                    except Exception:
                        pass
                return pd.concat(frames) if frames else None

            # Monthly zip: contains CSV/DAT directly
            return _parse_open_zip(zf, zip_path.name)

    except Exception as e:
        print(f"    Error reading {zip_path.name}: {e}")
        return None


def _parse_open_zip(zf, label):
    """Parse an already-open ZipFile containing a CSV or DAT data file."""
    import io as _io
    data_file = next(
        (n for n in zf.namelist() if n.upper().endswith(".CSV") or n.upper().endswith(".DAT")),
        None
    )
    if not data_file:
        return None

    raw = zf.read(data_file).decode("utf-8", errors="ignore")

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return None

    # Detect delimiter (comma or semicolon)
    sample = lines[0]
    delim = ";" if ";" in sample else ","

    rows = []
    for line in lines:
        parts = line.split(delim)
        if len(parts) >= 5:
            rows.append(parts)

    if not rows:
        return None

    # histdata formats:
    #   semicolon: "20240101 180000;open;high;low;close;vol"  (datetime in one field)
    #   comma:     "20240101,180000,open,high,low,close"       (date + time split)
    if delim == ";":
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df[0].str.strip(), format="%Y%m%d %H%M%S", errors="coerce")
        df = df.rename(columns={1: "open", 2: "high", 3: "low", 4: "close"})
    else:
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(
            df[0].str.strip() + " " + df[1].str.strip(),
            format="%Y%m%d %H%M%S",
            errors="coerce"
        )
        df = df.rename(columns={2: "open", 3: "high", 4: "low", 5: "close"})

    df = df.dropna(subset=["datetime"]).set_index("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close"]].dropna()


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
