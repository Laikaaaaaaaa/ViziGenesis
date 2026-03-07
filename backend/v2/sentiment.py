"""
ViziGenesis V2 — Sentiment Data Module
========================================
Ingests news, social, and fear/greed proxy signals:

Sources:
  • Yahoo Finance (yfinance) — built-in news sentiment
  • NewsAPI (if API key provided)
  • Reddit / social volume proxies
  • VIX-based fear/greed and put/call proxies

Produces four daily features:
  News_Sentiment       — rolling average headline sentiment ( −1 to +1 )
  Social_Sentiment     — social media momentum proxy
  Put_Call_Proxy       — VIX / realised-vol ratio
  Fear_Greed_Proxy     — composite fear & greed (0–100 scale)
"""
import logging, os, json, re, time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

logger = logging.getLogger("vizigenesis.v2.sentiment")


# ═══════════════════════════════════════════════════════════════════════
# 1.  Lightweight keyword sentiment scorer (no heavy NLP deps)
# ═══════════════════════════════════════════════════════════════════════
_POS_WORDS = {
    "surge", "soar", "rally", "gain", "profit", "beat", "exceed",
    "upgrade", "bullish", "outperform", "record", "strong", "growth",
    "positive", "boost", "breakthrough", "innovation", "optimistic",
    "recovery", "rebound", "upside", "up", "high", "buy", "win",
    "accelerate", "expand", "dividend", "earnings beat",
}
_NEG_WORDS = {
    "crash", "plunge", "decline", "loss", "miss", "downgrade",
    "bearish", "underperform", "weak", "risk", "fear", "warning",
    "negative", "cut", "layoff", "recession", "inflation", "sell",
    "drop", "fall", "down", "low", "crisis", "concern", "debt",
    "lawsuit", "slump", "default", "bankruptcy", "volatility",
}


def _keyword_sentiment(text: str) -> float:
    """Return sentiment score −1.0 to +1.0 from keyword matching."""
    if not text:
        return 0.0
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ═══════════════════════════════════════════════════════════════════════
# 2.  Yahoo Finance news sentiment (per-symbol)
# ═══════════════════════════════════════════════════════════════════════
def fetch_yahoo_news_sentiment(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch recent news for a symbol from Yahoo Finance and score sentiment.
    Returns DataFrame with columns: [date, sentiment, headline].
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
    except Exception as e:
        logger.warning("Yahoo news fetch for %s failed: %s", symbol, e)
        news = []

    records = []
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    for item in news:
        try:
            ts = item.get("providerPublishTime", 0)
            if ts == 0:
                pub = item.get("content", {}).get("pubDate", "")
                if pub:
                    dt = pd.Timestamp(pub)
                else:
                    continue
            else:
                dt = pd.Timestamp(ts, unit="s")

            if dt < pd.Timestamp(cutoff):
                continue

            title = item.get("title", "") or item.get("content", {}).get("title", "")
            summary = item.get("summary", "") or item.get("content", {}).get("summary", "")
            text = f"{title} {summary}"
            score = _keyword_sentiment(text)

            records.append({
                "date": dt.normalize(),
                "sentiment": score,
                "headline": title[:200],
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["date", "sentiment", "headline"])

    df = pd.DataFrame(records)
    return df


def compute_daily_news_sentiment(symbol: str, lookback_days: int = 365) -> pd.Series:
    """
    Return daily aggregated news sentiment for a symbol.
    Uses 5-day rolling average for smoothing.
    """
    df = fetch_yahoo_news_sentiment(symbol, lookback_days)
    if df.empty:
        return pd.Series(dtype=float, name="News_Sentiment")

    daily = df.groupby("date")["sentiment"].mean()
    daily = daily.sort_index()
    # Rolling 5-day average with forward fill
    smoothed = daily.rolling(5, min_periods=1).mean()
    smoothed.name = "News_Sentiment"
    return smoothed


# ═══════════════════════════════════════════════════════════════════════
# 3.  NewsAPI integration (optional — requires API key)
# ═══════════════════════════════════════════════════════════════════════
def fetch_newsapi_sentiment(
    query: str,
    api_key: Optional[str] = None,
    lookback_days: int = 30,
) -> pd.Series:
    """Fetch news from NewsAPI and compute daily sentiment."""
    key = api_key or os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return pd.Series(dtype=float, name="News_Sentiment")

    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={from_date}&language=en&sortBy=publishedAt&apiKey={key}"
    )
    try:
        req = Request(url, headers={"User-Agent": "ViziGenesis/2.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        articles = data.get("articles", [])
    except Exception as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return pd.Series(dtype=float, name="News_Sentiment")

    records = []
    for art in articles:
        title = art.get("title", "")
        desc = art.get("description", "")
        pub = art.get("publishedAt", "")
        if not pub:
            continue
        dt = pd.Timestamp(pub).normalize()
        score = _keyword_sentiment(f"{title} {desc}")
        records.append({"date": dt, "sentiment": score})

    if not records:
        return pd.Series(dtype=float, name="News_Sentiment")

    df = pd.DataFrame(records)
    daily = df.groupby("date")["sentiment"].mean().sort_index()
    return daily.rolling(3, min_periods=1).mean().rename("News_Sentiment")


# ═══════════════════════════════════════════════════════════════════════
# 4.  Social sentiment proxy (Reddit-like volume signals)
# ═══════════════════════════════════════════════════════════════════════
def compute_social_sentiment_proxy(
    stock_close: pd.Series,
    vix: pd.Series,
) -> pd.Series:
    """
    Proxy for social sentiment momentum using volume surprise + VIX.
    In production, replace with actual Reddit/Twitter API data.

    Logic: high-volume days with positive returns = positive social momentum;
    high-volume with negative returns + high VIX = negative social.
    """
    if stock_close.empty:
        return pd.Series(dtype=float, name="Social_Sentiment")

    ret = stock_close.pct_change()
    # Momentum of returns as social proxy
    momentum = ret.rolling(5, min_periods=1).mean()

    # VIX-adjusted: negative sentiment amplified when VIX high
    vix_aligned = vix.reindex(stock_close.index).ffill().bfill()
    vix_zscore = (vix_aligned - vix_aligned.rolling(60, min_periods=10).mean()) / \
                 vix_aligned.rolling(60, min_periods=10).std().clip(lower=1)

    social = momentum - 0.3 * vix_zscore.clip(-2, 2)
    social = social.clip(-1, 1)
    social.name = "Social_Sentiment"
    return social.fillna(0)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Fear / Greed proxy & Put/Call proxy
# ═══════════════════════════════════════════════════════════════════════
def compute_fear_greed_proxy(
    sp500_ret: pd.Series,
    vix: pd.Series,
    bond_10y: pd.Series,
) -> pd.Series:
    """
    Composite fear/greed proxy (0–100 scale, 50 = neutral).
    Components:
      • VIX level (inverted — high VIX = fear)
      • Market momentum (SP500 20-day return)
      • Bond yield change (flight to safety proxy)
    """
    vix_norm = vix.rolling(252, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / max(x.max() - x.min(), 0.01), raw=False
    )
    vix_component = (1 - vix_norm) * 100  # 0=extreme fear, 100=greed

    mkt_mom = sp500_ret.rolling(20, min_periods=5).sum()
    mkt_norm = mkt_mom.rolling(252, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / max(x.max() - x.min(), 0.01), raw=False
    ) * 100

    yield_chg = bond_10y.diff(5)
    yield_norm = yield_chg.rolling(252, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / max(x.max() - x.min(), 0.01), raw=False
    )
    bond_component = (1 - yield_norm) * 100  # falling yields = fear

    composite = (
        0.40 * vix_component.fillna(50) +
        0.35 * mkt_norm.fillna(50) +
        0.25 * bond_component.fillna(50)
    )
    composite = composite.clip(0, 100)
    composite.name = "Fear_Greed_Proxy"
    return composite.fillna(50)


def compute_put_call_proxy(vix: pd.Series, realised_vol: pd.Series) -> pd.Series:
    """
    Put/Call ratio proxy = VIX / Realised_Vol.
    When VIX >> realised vol → market is pricing in more downside than recent history.
    """
    rv = realised_vol.clip(lower=0.01)
    ratio = vix / rv
    ratio = ratio.clip(0.5, 3.0)
    ratio.name = "Put_Call_Proxy"
    return ratio.fillna(1.0)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Build all sentiment features for a stock
# ═══════════════════════════════════════════════════════════════════════
def build_sentiment_features(
    stock_df: pd.DataFrame,
    market_context: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Build all 4 sentiment features aligned to stock_df index:
      News_Sentiment, Social_Sentiment, Put_Call_Proxy, Fear_Greed_Proxy
    """
    idx = stock_df.index

    # 1) News sentiment
    try:
        news_sent = compute_daily_news_sentiment(symbol, lookback_days=365)
        news_sent = news_sent.reindex(idx).ffill().bfill().fillna(0)
    except Exception as e:
        logger.warning("News sentiment failed for %s: %s", symbol, e)
        news_sent = pd.Series(0.0, index=idx, name="News_Sentiment")

    # 2) Social sentiment proxy
    vix = market_context["VIX"].reindex(idx).ffill().bfill() if "VIX" in market_context.columns else pd.Series(20.0, index=idx)
    social_sent = compute_social_sentiment_proxy(stock_df["Close"], vix)
    social_sent = social_sent.reindex(idx).fillna(0)

    # 3) Put/Call proxy
    rv_20 = stock_df["Close"].pct_change().rolling(20, min_periods=5).std() * np.sqrt(252) * 100
    put_call = compute_put_call_proxy(vix, rv_20)
    put_call = put_call.reindex(idx).fillna(1.0)

    # 4) Fear/Greed proxy
    sp500_ret = market_context["SP500_Ret"].reindex(idx).ffill().fillna(0) if "SP500_Ret" in market_context.columns else pd.Series(0.0, index=idx)
    bond_10y = market_context["BOND_10Y"].reindex(idx).ffill().fillna(3.0) if "BOND_10Y" in market_context.columns else pd.Series(3.0, index=idx)
    fear_greed = compute_fear_greed_proxy(sp500_ret, vix, bond_10y)
    fear_greed = fear_greed.reindex(idx).fillna(50)

    return pd.DataFrame({
        "News_Sentiment": news_sent,
        "Social_Sentiment": social_sent,
        "Put_Call_Proxy": put_call,
        "Fear_Greed_Proxy": fear_greed,
    }, index=idx)
