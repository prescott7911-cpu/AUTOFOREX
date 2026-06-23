# AUTOFOREX

An automated forex trading toolkit.

## Features

- EMA crossover + pullback + engulfing candle signal detection
- 3-way position split exits at previous day liquidity levels
- Per-trade risk sizing (2.5% default) with daily drawdown limit
- Equity curve, P&L, and drawdown charts

## Getting Started

```bash
pip install -r requirements.txt
```

**Run the backtest:**
```bash
python backtest.py
```

Place your `XAUUSD_M15.csv` file in this directory first. Expected columns: `datetime, open, high, low, close`.

Free data sources:
- https://www.histdata.com
- https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed/

## Requirements

- Python 3.10+
- A broker API key (e.g. OANDA, Interactive Brokers)

## License

MIT
