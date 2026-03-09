#!/usr/bin/env python3
"""
ViziGenesis — Data Preprocessing Pipeline
==========================================
Standardizes all collected datasets and prepares them for LLM training:

1. Standardize CSV formats (consistent date index, column names)
2. Label datasets by type (macro, stock, commodity, news, transcript)
3. Tokenize / preprocess text data (news, transcripts, central bank)
4. Build unified training corpus in JSONL format

Run:  python data/preprocess.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

# ══════════════════════════════════════════════════════════════
#  1. CSV STANDARDIZATION
# ══════════════════════════════════════════════════════════════

def standardize_csv(path: Path, dataset_type: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV and standardize:
    - Ensure datetime index (named 'date')
    - Numeric columns coerced
    - Drop all-NaN rows
    - Add metadata columns
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    # Detect date column
    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "datetime", "time", "timestamp", "index"):
            date_col = col
            break
    if date_col is None and len(df.columns) >= 1:
        # Try first column
        try:
            pd.to_datetime(df.iloc[:, 0], errors="raise")
            date_col = df.columns[0]
        except Exception:
            pass

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index.name = "date"

    # Coerce numeric
    for col in df.columns:
        if df[col].dtype == object:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > len(df) * 0.5:
                df[col] = numeric

    df = df.dropna(how="all")
    return df


def standardize_all_csvs() -> Dict[str, int]:
    """Walk all data directories and standardize CSV files."""
    print("\n=== STANDARDIZING CSV FILES ===")
    stats: Dict[str, int] = {}

    type_dirs = {
        "macro": DATA / "macro",
        "markets": DATA / "markets",
        "stocks": DATA / "stocks",
        "fundamentals": DATA / "fundamentals",
    }

    (PROCESSED / "summaries").mkdir(parents=True, exist_ok=True)

    for dtype, base_dir in type_dirs.items():
        if not base_dir.exists():
            continue
        csv_files = list(base_dir.rglob("*.csv"))
        count = 0
        for csv_path in csv_files:
            df = standardize_csv(csv_path, dtype)
            if df is not None and not df.empty:
                count += 1
        stats[dtype] = count
        print(f"  {dtype}: {count} CSV files standardized")

    return stats


# ══════════════════════════════════════════════════════════════
#  2. DATASET LABELING & SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════

def generate_dataset_catalog() -> List[Dict[str, Any]]:
    """
    Generate a catalog of all datasets with metadata.
    This is used both for documentation and for the model to understand
    what data is available.
    """
    print("\n=== GENERATING DATASET CATALOG ===")
    catalog: List[Dict[str, Any]] = []

    # Macro datasets
    macro_dir = DATA / "macro"
    if macro_dir.exists():
        for csv_path in sorted(macro_dir.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path, nrows=5)
                rows = sum(1 for _ in open(csv_path, encoding="utf-8", errors="replace")) - 1
            except Exception:
                continue
            rel = csv_path.relative_to(DATA)
            catalog.append({
                "path": str(rel),
                "type": "macro",
                "sub_type": csv_path.parent.name,
                "name": csv_path.stem,
                "columns": list(df.columns),
                "rows": max(rows, 0),
                "format": "csv",
            })

    # Market datasets
    markets_dir = DATA / "markets"
    if markets_dir.exists():
        for csv_path in sorted(markets_dir.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path, nrows=5)
                rows = sum(1 for _ in open(csv_path, encoding="utf-8", errors="replace")) - 1
            except Exception:
                continue
            rel = csv_path.relative_to(DATA)
            catalog.append({
                "path": str(rel),
                "type": "markets",
                "sub_type": csv_path.parent.name,
                "name": csv_path.stem,
                "columns": list(df.columns),
                "rows": max(rows, 0),
                "format": "csv",
            })

    # Stock datasets
    stocks_dir = DATA / "stocks"
    if stocks_dir.exists():
        for sym_dir in sorted(stocks_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            ohlcv = sym_dir / "ohlcv.csv"
            if ohlcv.exists():
                try:
                    rows = sum(1 for _ in open(ohlcv, encoding="utf-8", errors="replace")) - 1
                except Exception:
                    rows = 0
                files = [f.name for f in sym_dir.iterdir() if f.is_file()]
                catalog.append({
                    "path": str(sym_dir.relative_to(DATA)),
                    "type": "stock",
                    "name": sym_dir.name,
                    "rows": max(rows, 0),
                    "files": files,
                    "format": "csv",
                })

    # Fundamentals
    fund_dir = DATA / "fundamentals"
    if fund_dir.exists():
        for sym_dir in sorted(fund_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            files = [f.name for f in sym_dir.iterdir() if f.is_file()]
            catalog.append({
                "path": str(sym_dir.relative_to(DATA)),
                "type": "fundamentals",
                "name": sym_dir.name,
                "files": files,
                "format": "csv+json",
            })

    # News
    news_dir = DATA / "news"
    if news_dir.exists():
        for jsonl_path in sorted(news_dir.rglob("*.jsonl")):
            try:
                lines = sum(1 for _ in open(jsonl_path, encoding="utf-8", errors="replace"))
            except Exception:
                lines = 0
            catalog.append({
                "path": str(jsonl_path.relative_to(DATA)),
                "type": "news",
                "name": jsonl_path.stem,
                "rows": lines,
                "format": "jsonl",
            })

    # Transcripts
    trans_dir = DATA / "transcripts"
    if trans_dir.exists():
        for sym_dir in sorted(trans_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            files = list(sym_dir.glob("*.json"))
            catalog.append({
                "path": str(sym_dir.relative_to(DATA)),
                "type": "transcript",
                "name": sym_dir.name,
                "files": [f.name for f in files],
                "count": len(files),
                "format": "json",
            })

    # Central bank
    cb_dir = DATA / "central_bank"
    if cb_dir.exists():
        for jsonl_path in sorted(cb_dir.rglob("*.jsonl")):
            try:
                lines = sum(1 for _ in open(jsonl_path, encoding="utf-8", errors="replace"))
            except Exception:
                lines = 0
            catalog.append({
                "path": str(jsonl_path.relative_to(DATA)),
                "type": "central_bank",
                "name": jsonl_path.stem,
                "rows": lines,
                "format": "jsonl",
            })

    # Save
    (PROCESSED).mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "dataset_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"  Catalog: {len(catalog)} datasets indexed")
    return catalog


# ══════════════════════════════════════════════════════════════
#  3. TEXT PREPROCESSING (news, transcripts, central bank)
# ══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean and normalize text for LLM training."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_news() -> int:
    """
    Process all news JSONL files into a clean training corpus.
    Output: processed/news_corpus.jsonl
    """
    print("\n=== PREPROCESSING NEWS ===")
    rss_dir = DATA / "news" / "rss"
    if not rss_dir.exists():
        print("  No news data found")
        return 0

    total = 0
    out_path = PROCESSED / "news_corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for jsonl_file in sorted(rss_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        art = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = clean_text(art.get("title", ""))
                    desc = clean_text(art.get("description", "") or art.get("summary", ""))
                    date = art.get("date", "")
                    source = art.get("source", jsonl_file.stem)

                    if not title:
                        continue

                    processed = {
                        "type": "news",
                        "source": source,
                        "date": date,
                        "title": title,
                        "content": desc,
                        "text": f"[NEWS] [{source}] [{date}] {title}. {desc}".strip(),
                    }
                    out.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    total += 1

    print(f"  Processed {total} news articles → {out_path.relative_to(ROOT)}")
    return total


def preprocess_central_bank() -> int:
    """Process central bank data into training corpus."""
    print("\n=== PREPROCESSING CENTRAL BANK DATA ===")
    cb_dir = DATA / "central_bank"
    if not cb_dir.exists():
        print("  No central bank data found")
        return 0

    total = 0
    out_path = PROCESSED / "central_bank_corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for jsonl_file in sorted(cb_dir.rglob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = clean_text(item.get("title", ""))
                    desc = clean_text(item.get("description", "") or item.get("summary", ""))
                    date = item.get("date", "")
                    source = item.get("source", "fed")

                    if not title:
                        continue

                    processed = {
                        "type": "central_bank",
                        "source": source,
                        "date": date,
                        "title": title,
                        "content": desc,
                        "text": f"[CENTRAL_BANK] [{source}] [{date}] {title}. {desc}".strip(),
                    }
                    out.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    total += 1

    print(f"  Processed {total} central bank items → {out_path.relative_to(ROOT)}")
    return total


def preprocess_transcripts() -> int:
    """Process SEC filing metadata into training corpus."""
    print("\n=== PREPROCESSING SEC FILINGS ===")
    trans_dir = DATA / "transcripts"
    if not trans_dir.exists():
        print("  No transcript data found")
        return 0

    total = 0
    out_path = PROCESSED / "filings_corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for sym_dir in sorted(trans_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            ticker = sym_dir.name
            for json_file in sorted(sym_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    continue

                form = meta.get("form", "")
                filing_date = meta.get("filing_date", "")

                processed = {
                    "type": "sec_filing",
                    "ticker": ticker,
                    "form": form,
                    "filing_date": filing_date,
                    "url": meta.get("url", ""),
                    "text": f"[SEC_FILING] {ticker} filed {form} on {filing_date}.",
                }
                out.write(json.dumps(processed, ensure_ascii=False) + "\n")
                total += 1

    print(f"  Processed {total} SEC filings → {out_path.relative_to(ROOT)}")
    return total


# ══════════════════════════════════════════════════════════════
#  4. BUILD NUMERIC DATA SUMMARIES FOR LLM
# ══════════════════════════════════════════════════════════════

def build_macro_summaries() -> int:
    """
    Convert macro CSV time series into natural language summaries
    that the model can learn from.
    """
    print("\n=== BUILDING MACRO SUMMARIES ===")
    macro_dir = DATA / "macro" / "fred"
    if not macro_dir.exists():
        print("  No FRED data found")
        return 0

    total = 0
    out_path = PROCESSED / "macro_summaries.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for csv_path in sorted(macro_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
                if df.empty or len(df.columns) < 2:
                    continue
                df.columns = ["date", "value"]
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna().sort_values("date")
            except Exception:
                continue

            name = csv_path.stem
            if len(df) < 5:
                continue

            # Generate summary text
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            first = df.iloc[0]

            latest_val = latest["value"]
            prev_val = prev["value"]
            change = latest_val - prev_val
            pct_change = (change / prev_val * 100) if prev_val != 0 else 0

            summary = {
                "type": "macro_summary",
                "indicator": name,
                "latest_date": str(latest["date"].date()) if hasattr(latest["date"], "date") else str(latest["date"]),
                "latest_value": round(latest_val, 4),
                "previous_value": round(prev_val, 4),
                "change": round(change, 4),
                "pct_change": round(pct_change, 2),
                "history_start": str(first["date"].date()) if hasattr(first["date"], "date") else str(first["date"]),
                "data_points": len(df),
                "text": (
                    f"[MACRO] {name.replace('_', ' ').title()}: "
                    f"Latest value is {latest_val:.4g} as of {latest['date']:%Y-%m-%d}. "
                    f"Previous was {prev_val:.4g}, change of {change:+.4g} ({pct_change:+.2f}%). "
                    f"Data available from {first['date']:%Y-%m-%d} ({len(df)} observations)."
                ),
            }
            out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            total += 1

    print(f"  Generated {total} macro summaries → {out_path.relative_to(ROOT)}")
    return total


def build_stock_summaries() -> int:
    """Convert stock data into natural language summaries."""
    print("\n=== BUILDING STOCK SUMMARIES ===")
    stocks_dir = DATA / "stocks"
    if not stocks_dir.exists():
        print("  No stock data found")
        return 0

    total = 0
    out_path = PROCESSED / "stock_summaries.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for sym_dir in sorted(stocks_dir.iterdir()):
            if not sym_dir.is_dir():
                continue

            ohlcv_path = sym_dir / "ohlcv.csv"
            if not ohlcv_path.exists():
                continue

            try:
                df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
                if df.empty:
                    continue
            except Exception:
                continue

            sym = sym_dir.name
            c = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
            v = pd.to_numeric(df.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()

            if len(c) < 10:
                continue

            latest_price = c.iloc[-1]
            prev_price = c.iloc[-2]
            daily_ret = (latest_price / prev_price - 1) * 100

            # Period returns
            rets = {}
            for days, label in [(5, "1w"), (21, "1m"), (63, "3m"), (126, "6m"), (252, "1y")]:
                if len(c) > days:
                    rets[label] = round((c.iloc[-1] / c.iloc[-days] - 1) * 100, 2)

            high_52w = c.iloc[-min(252, len(c)):].max()
            low_52w = c.iloc[-min(252, len(c)):].min()
            avg_vol = int(v.iloc[-20:].mean()) if len(v) >= 20 else 0

            summary = {
                "type": "stock_summary",
                "symbol": sym,
                "latest_date": str(c.index[-1].date()) if hasattr(c.index[-1], "date") else str(c.index[-1]),
                "latest_price": round(latest_price, 2),
                "daily_return_pct": round(daily_ret, 2),
                "returns": rets,
                "52w_high": round(high_52w, 2),
                "52w_low": round(low_52w, 2),
                "avg_volume_20d": avg_vol,
                "history_days": len(c),
                "text": (
                    f"[STOCK] {sym}: Price ${latest_price:.2f}, daily {daily_ret:+.2f}%. "
                    f"Returns: {', '.join(f'{k}={v:+.1f}%' for k, v in rets.items())}. "
                    f"52-week range: ${low_52w:.2f}-${high_52w:.2f}. "
                    f"Avg volume (20d): {avg_vol:,}. History: {len(c)} days."
                ),
            }
            out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            total += 1

    print(f"  Generated {total} stock summaries → {out_path.relative_to(ROOT)}")
    return total


def build_fundamentals_summaries() -> int:
    """Convert fundamentals data into natural language summaries."""
    print("\n=== BUILDING FUNDAMENTALS SUMMARIES ===")
    fund_dir = DATA / "fundamentals"
    if not fund_dir.exists():
        print("  No fundamentals data found")
        return 0

    total = 0
    out_path = PROCESSED / "fundamentals_summaries.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for sym_dir in sorted(fund_dir.iterdir()):
            if not sym_dir.is_dir():
                continue

            sym = sym_dir.name
            stats_path = sym_dir / "key_stats.json"
            if not stats_path.exists():
                continue

            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                continue

            # Build text description
            parts = [f"[FUNDAMENTALS] {sym}:"]

            if "sector" in info:
                parts.append(f"Sector: {info['sector']}, Industry: {info.get('industry', 'N/A')}.")

            if "marketCap" in info:
                mc = info["marketCap"]
                if mc > 1e12:
                    parts.append(f"Market Cap: ${mc/1e12:.1f}T.")
                elif mc > 1e9:
                    parts.append(f"Market Cap: ${mc/1e9:.1f}B.")
                else:
                    parts.append(f"Market Cap: ${mc/1e6:.0f}M.")

            if "trailingPE" in info:
                parts.append(f"P/E: {info['trailingPE']:.1f} (trailing), {info.get('forwardPE', 'N/A')} (forward).")

            if "profitMargins" in info:
                parts.append(f"Margins: profit={info['profitMargins']*100:.1f}%, "
                           f"operating={info.get('operatingMargins', 0)*100:.1f}%, "
                           f"gross={info.get('grossMargins', 0)*100:.1f}%.")

            if "returnOnEquity" in info:
                parts.append(f"ROE: {info['returnOnEquity']*100:.1f}%, ROA: {info.get('returnOnAssets', 0)*100:.1f}%.")

            if "totalDebt" in info:
                d = info["totalDebt"]
                parts.append(f"Total Debt: ${d/1e9:.1f}B, D/E: {info.get('debtToEquity', 'N/A')}.")

            if "freeCashflow" in info:
                fcf = info["freeCashflow"]
                parts.append(f"Free Cash Flow: ${fcf/1e9:.1f}B.")

            if "trailingEps" in info:
                parts.append(f"EPS: {info['trailingEps']:.2f} (trailing), {info.get('forwardEps', 'N/A')} (forward).")

            if "revenueGrowth" in info:
                parts.append(f"Revenue Growth: {info['revenueGrowth']*100:.1f}%.")

            if "dividendYield" in info and info["dividendYield"]:
                parts.append(f"Dividend Yield: {info['dividendYield']*100:.2f}%.")

            if "beta" in info:
                parts.append(f"Beta: {info['beta']:.2f}.")

            summary = {
                "type": "fundamentals_summary",
                "symbol": sym,
                "data": info,
                "text": " ".join(parts),
            }
            out.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
            total += 1

    print(f"  Generated {total} fundamentals summaries → {out_path.relative_to(ROOT)}")
    return total


# ══════════════════════════════════════════════════════════════
#  5. CROSS-DATASET CORRELATION NARRATIVES
# ══════════════════════════════════════════════════════════════

def build_correlation_narratives() -> int:
    """
    Build narratives about macro→market correlations for LLM training.
    Example: "When CPI rises, SP500 tends to fall because..."
    """
    print("\n=== BUILDING CORRELATION NARRATIVES ===")

    # Template-based knowledge injection
    narratives = [
        # Inflation → Markets
        {
            "type": "correlation_narrative",
            "topic": "CPI_vs_SP500",
            "text": (
                "[ANALYSIS] CPI and S&P 500 Correlation: When CPI (Consumer Price Index) rises "
                "above expectations, the S&P 500 tends to decline as investors price in tighter "
                "monetary policy from the Fed. Higher inflation → higher interest rates → lower "
                "equity valuations (higher discount rates). Growth stocks with distant cash flows "
                "are most sensitive. Historically, a 0.1% CPI surprise above consensus correlates "
                "with -0.5% to -1.5% same-day SP500 move."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "FedFunds_vs_Markets",
            "text": (
                "[ANALYSIS] Fed Funds Rate and Market Impact: The Federal Funds Rate is the primary "
                "tool the Fed uses to control monetary policy. Rate hikes increase borrowing costs, "
                "slow economic growth, and reduce corporate earnings — negative for stocks. Rate cuts "
                "stimulate the economy and are generally positive for equities. However, the market's "
                "reaction depends on expectations: if a rate cut was already priced in, the market may "
                "not move. The 'dot plot' and forward guidance matter as much as the actual decision."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "YieldCurve_Recession",
            "text": (
                "[ANALYSIS] Yield Curve Inversion and Recession: When the 10Y-2Y Treasury spread "
                "goes negative (inverts), it has historically preceded recessions within 12-24 months "
                "with high accuracy. An inverted yield curve means short-term rates exceed long-term "
                "rates, signaling the market expects the Fed will need to cut rates due to economic "
                "weakness. The un-inversion (steepening after inversion) is often the more immediate "
                "signal that recession is near."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "VIX_Market_Stress",
            "text": (
                "[ANALYSIS] VIX and Market Stress: The VIX (CBOE Volatility Index) measures expected "
                "30-day volatility of the S&P 500. VIX below 15 indicates complacency, 15-25 is normal, "
                "25-35 indicates elevated fear, above 35 signals panic. VIX spikes often coincide with "
                "market sell-offs. Mean-reverting property: very high VIX readings (>40) often mark "
                "market bottoms. The VIX term structure (contango vs backwardation) provides additional "
                "information about market stress levels."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "DXY_EM_Stocks",
            "text": (
                "[ANALYSIS] US Dollar (DXY) and Emerging Markets: A strong US dollar (rising DXY) is "
                "typically negative for emerging market stocks and commodities. EM countries with USD-"
                "denominated debt face higher repayment costs. Capital flows from EM to US as higher "
                "US rates attract investment. Commodities priced in USD become more expensive for "
                "foreign buyers. Conversely, a weakening dollar benefits EM assets and commodities."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Oil_Inflation_Earnings",
            "text": (
                "[ANALYSIS] Oil Prices and the Economy: Rising oil prices act as a tax on consumers "
                "and businesses, increasing production costs and reducing disposable income. Energy "
                "sector benefits directly (XOM, CVX). Airlines, shipping, and manufacturing are hurt. "
                "Very high oil prices (>$100/bbl) historically precede economic slowdowns. Oil price "
                "spikes of 50%+ in a year are associated with recession risk."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Gold_SafeHaven",
            "text": (
                "[ANALYSIS] Gold as Safe Haven: Gold tends to rise during geopolitical uncertainty, "
                "financial crises, and high inflation. It's traditionally inversely correlated with "
                "real yields (TIPS yields). When real yields fall (negative real rates), gold becomes "
                "more attractive as opportunity cost of holding zero-yield gold decreases. Central bank "
                "gold purchases (China, Russia, Turkey) have been a major demand driver since 2022."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Unemployment_Stocks",
            "text": (
                "[ANALYSIS] Unemployment Rate and Stocks: Rising unemployment is a lagging indicator — "
                "it peaks after recessions. However, the pace of change matters more than the level. "
                "A rapid rise in unemployment (Sahm Rule: 0.5% increase from 12-month low) reliably "
                "signals recession onset. Markets often fall before unemployment rises and recover "
                "before it peaks. Strong labor market supports consumer spending and corporate earnings."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Earnings_StockPrice",
            "text": (
                "[ANALYSIS] Earnings and Stock Prices: Earnings season is a major catalyst for "
                "individual stocks. Key metrics: EPS beat/miss vs consensus, revenue growth, margins, "
                "forward guidance. A company beating EPS estimates by 5%+ with strong guidance typically "
                "sees 3-5% gains. Missing estimates or lowering guidance causes 5-10% declines. The "
                "market context matters: in a risk-off environment, even good earnings may not help."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "PCE_FedPolicy",
            "text": (
                "[ANALYSIS] PCE and Fed Policy: The Personal Consumption Expenditures Price Index "
                "(especially Core PCE) is the Fed's preferred inflation measure. The Fed targets 2% "
                "Core PCE. Above 2% → hawkish (higher rates), below → dovish (easier policy). The gap "
                "between Core PCE and CPI can create confusion. PCE gives more weight to healthcare and "
                "housing substitution effects. Market reactions to PCE data are among the strongest."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "China_GlobalTrade",
            "text": (
                "[ANALYSIS] China Economy and Global Impact: China's GDP growth directly impacts global "
                "commodity demand (copper, iron ore, oil). Weak Chinese data (PMI, exports, property) "
                "tends to drag global markets. China's stimulus measures (PBOC rate cuts, RRR cuts, "
                "fiscal spending) are positive for commodities and EM. Geopolitical tensions (US-China "
                "trade war, Taiwan) create sector-specific risks in semiconductors and tech supply chains."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "CreditSpreads_Risk",
            "text": (
                "[ANALYSIS] Credit Spreads and Financial Risk: The HY OAS (High-Yield Option-Adjusted "
                "Spread) measures credit risk pricing. Spreads below 300bp = risk complacency, "
                "300-500bp = normal, 500-800bp = stress, above 800bp = crisis. Rapidly widening spreads "
                "precede equity market declines and signal corporate default risk. Investment-grade (IG) "
                "spreads are a leading indicator for economic conditions."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "TechEarnings_Nasdaq",
            "text": (
                "[ANALYSIS] Big Tech Earnings and NASDAQ: The 'Magnificent 7' (AAPL, MSFT, GOOGL, "
                "AMZN, META, NVDA, TSLA) dominate the NASDAQ 100 and S&P 500 by market weight. "
                "Their earnings reports can move the entire market. AI/cloud revenue growth (MSFT Azure, "
                "GOOGL Cloud, AMZN AWS) is a key thematic driver. Capex spending on AI infrastructure "
                "benefits NVDA, AVGO, AMD. Advertising revenue trends affect META, GOOGL, SNAP."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "QE_QT_Liquidity",
            "text": (
                "[ANALYSIS] Quantitative Easing/Tightening and Liquidity: The Fed's balance sheet "
                "size (WALCL) directly affects market liquidity. QE (asset purchases) injects liquidity → "
                "positive for risk assets. QT (balance sheet reduction) drains liquidity → headwind for "
                "stocks. Combined with the Treasury General Account (TGA) and Reverse Repo facility (RRP), "
                "net liquidity = Fed Balance Sheet - TGA - RRP. Rising net liquidity correlates strongly "
                "with S&P 500 performance."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Sector_Rotation",
            "text": (
                "[ANALYSIS] Sector Rotation Through Business Cycles: Early cycle (recovery): Consumer "
                "Discretionary, Financials, Industrials outperform. Mid cycle (expansion): Technology, "
                "Communication Services lead. Late cycle (slowdown): Energy, Materials, Healthcare. "
                "Recession: Utilities, Consumer Staples, Healthcare are defensive. Rate-sensitive sectors: "
                "Financials benefit from rising rates (wider NIM), REITs/Utilities hurt by rising rates."
            ),
        },
        {
            "type": "correlation_narrative",
            "topic": "Vietnam_Market",
            "text": (
                "[ANALYSIS] Vietnam Stock Market (VN-Index): The VN-Index is heavily weighted toward "
                "real estate (VIC, VHM), banking (VCB, TCB, CTG), and steel (HPG). Key drivers: "
                "State Bank of Vietnam policy rate, USD/VND exchange rate, foreign investment flows, "
                "China+1 manufacturing shift, real estate market health. Vietnam's market is sensitive "
                "to US rate decisions (capital flows) and China's economic performance (trade linkages)."
            ),
        },
    ]

    out_path = PROCESSED / "correlation_narratives.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for n in narratives:
            out.write(json.dumps(n, ensure_ascii=False) + "\n")

    print(f"  Generated {len(narratives)} correlation narratives → {out_path.relative_to(ROOT)}")
    return len(narratives)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"ViziGenesis Preprocessing Pipeline")
    print(f"Output: {PROCESSED}")
    print(f"{'='*60}")

    stats: Dict[str, Any] = {}

    stats["csv_standardized"] = standardize_all_csvs()
    stats["catalog"] = len(generate_dataset_catalog())
    stats["news_articles"] = preprocess_news()
    stats["central_bank_items"] = preprocess_central_bank()
    stats["sec_filings"] = preprocess_transcripts()
    stats["macro_summaries"] = build_macro_summaries()
    stats["stock_summaries"] = build_stock_summaries()
    stats["fundamentals_summaries"] = build_fundamentals_summaries()
    stats["correlation_narratives"] = build_correlation_narratives()

    with open(PROCESSED / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete. Stats: {json.dumps(stats, indent=2)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
