"""
Generate sample CSV data for offline testing.
Run:  python -m data.generate_sample
"""
import os, sys
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _safe_symbol(symbol: str) -> str:
    return (symbol or "").upper().replace("/", "_").replace("\\", "_").strip()


def symbol_data_dir(symbol: str) -> str:
    path = os.path.join(DATA_DIR, _safe_symbol(symbol))
    os.makedirs(path, exist_ok=True)
    return path


def generate_synthetic_stock(symbol: str = "SAMPLE", days: int = 500,
                              start_price: float = 150.0):
    """Create a realistic-looking synthetic OHLCV dataset."""
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    prices = [start_price]
    for _ in range(days - 1):
        change = np.random.normal(0, 0.018) * prices[-1]
        prices.append(max(prices[-1] + change, 1.0))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.008, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, days)))
    opn = low + (high - low) * np.random.uniform(0.3, 0.7, days)
    volume = np.random.randint(10_000_000, 100_000_000, days)

    df = pd.DataFrame({
        "Date": dates[:days],
        "Open": np.round(opn, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume,
    })
    df.set_index("Date", inplace=True)

    path = os.path.join(symbol_data_dir(symbol), "sample.csv")
    df.to_csv(path)
    print(f"✅ Saved {days} rows → {path}")
    return path


if __name__ == "__main__":
    generate_synthetic_stock("AAPL", 500, 150.0)
    generate_synthetic_stock("TSLA", 500, 220.0)
    generate_synthetic_stock("MSFT", 500, 340.0)
    generate_synthetic_stock("VIC.VN", 500, 58.0)
    generate_synthetic_stock("VNM.VN", 500, 72.0)
    print("Done — sample data is in /data/<SYMBOL>/sample.csv")
