#!/usr/bin/env python3
"""
ViziGenesis — RAG (Retrieval-Augmented Generation) Engine
=========================================================
Provides real-time context retrieval from collected financial data
to ground the LLM's responses in actual data.

The RAG pipeline:
1. User asks a question
2. Retrieve relevant data snippets (macro, stock, news, transcripts)
3. Inject context into the LLM prompt
4. Generate grounded response

This works with both the fine-tuned model AND any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


class FinancialRAG:
    """
    Retrieval engine that searches local financial datasets
    for context relevant to a user query.
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or DATA
        self._macro_cache: Dict[str, pd.DataFrame] = {}
        self._stock_cache: Dict[str, pd.DataFrame] = {}
        self._fundamentals_cache: Dict[str, Dict] = {}
        self._catalog: List[Dict] = []
        self._news: List[Dict] = []
        self._narratives: List[Dict] = []
        self._loaded = False

    def load(self) -> None:
        """Load all data indexes into memory."""
        if self._loaded:
            return

        # Load dataset catalog
        catalog_path = self.data_root / "processed" / "dataset_catalog.json"
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                self._catalog = json.load(f)

        # Load macro summaries
        macro_path = self.data_root / "processed" / "macro_summaries.jsonl"
        if macro_path.exists():
            with open(macro_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self._macro_cache[item["indicator"]] = item
                    except Exception:
                        pass

        # Load stock summaries
        stock_path = self.data_root / "processed" / "stock_summaries.jsonl"
        if stock_path.exists():
            with open(stock_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self._stock_cache[item["symbol"]] = item
                    except Exception:
                        pass

        # Load fundamentals
        fund_path = self.data_root / "processed" / "fundamentals_summaries.jsonl"
        if fund_path.exists():
            with open(fund_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self._fundamentals_cache[item["symbol"]] = item
                    except Exception:
                        pass

        # Load news
        news_path = self.data_root / "processed" / "news_corpus.jsonl"
        if news_path.exists():
            with open(news_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self._news.append(json.loads(line))
                    except Exception:
                        pass

        # Load correlation narratives
        narr_path = self.data_root / "processed" / "correlation_narratives.jsonl"
        if narr_path.exists():
            with open(narr_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self._narratives.append(json.loads(line))
                    except Exception:
                        pass

        self._loaded = True
        total = (len(self._macro_cache) + len(self._stock_cache) +
                 len(self._fundamentals_cache) + len(self._news) + len(self._narratives))
        print(f"RAG loaded: {total} data points from {len(self._catalog)} datasets")

    def _extract_tickers(self, query: str) -> List[str]:
        """Extract stock ticker symbols from a query."""
        # Common patterns: $AAPL, AAPL, aapl
        tickers = re.findall(r'\$?([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b', query.upper())
        # Filter out common words that look like tickers
        noise = {"THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HOW",
                 "WHY", "CAN", "DID", "ALL", "ANY", "FEW", "OUR", "TWO", "OWN",
                 "TOP", "BIG", "LOW", "NEW", "OLD", "MAY", "NOW", "SET", "WAY",
                 "WHO", "OIL", "GAS", "FED", "GDP", "CPI", "PCE", "PMI", "IPO",
                 "ETF", "CEO", "CFO", "EPS", "ROE", "ROA", "PE", "BPS", "YOY",
                 "FX", "IF", "IT", "AT", "UP", "IS", "IN", "ON", "OR", "SO",
                 "NO", "DO", "TO"}
        return [t for t in tickers if t not in noise and t in self._stock_cache]

    def _extract_macro_keywords(self, query: str) -> List[str]:
        """Extract macro indicator keywords from query."""
        query_lower = query.lower()
        matches = []

        keyword_map = {
            "cpi": ["cpi_all_items", "cpi_core_less_food_energy"],
            "inflation": ["cpi_all_items", "core_pce_price_index", "breakeven_inflation_5y"],
            "pce": ["core_pce_price_index", "pce_price_index"],
            "unemployment": ["unemployment_rate"],
            "jobs": ["nonfarm_payrolls", "unemployment_rate", "initial_claims"],
            "payroll": ["nonfarm_payrolls"],
            "gdp": ["gdp_real", "gdp_real_growth_annualized"],
            "interest rate": ["fed_funds_rate", "effective_fed_funds_rate"],
            "fed funds": ["fed_funds_rate", "effective_fed_funds_rate"],
            "treasury": ["treasury_yield_10y", "treasury_yield_2y", "treasury_yield_30y"],
            "yield curve": ["term_spread_10y_2y", "term_spread_10y_3m"],
            "vix": ["vix_close"],
            "volatility": ["vix_close"],
            "dollar": ["trade_weighted_usd_broad"],
            "money supply": ["m2_money_stock", "m1_money_stock"],
            "balance sheet": ["fed_balance_sheet_total_assets"],
            "credit spread": ["hy_oas_spread", "baa_spread_over_10y"],
            "housing": ["housing_starts", "case_shiller_home_price_index", "mortgage_rate_30y"],
            "mortgage": ["mortgage_rate_30y", "mortgage_rate_15y"],
            "consumer": ["umich_consumer_sentiment", "retail_sales_ex_food_services"],
            "retail": ["retail_sales_ex_food_services"],
            "oil": ["wti_crude_spot", "brent_crude_spot"],
            "gold": ["gold_london_fix_pm"],
            "trade": ["trade_balance"],
            "claims": ["initial_claims", "continued_claims"],
        }

        for keyword, indicators in keyword_map.items():
            if keyword in query_lower:
                matches.extend(indicators)

        return list(set(matches))

    def _extract_narrative_topics(self, query: str) -> List[str]:
        """Find relevant correlation narratives."""
        query_lower = query.lower()
        topic_keywords = {
            "CPI_vs_SP500": ["cpi", "inflation", "sp500", "s&p"],
            "FedFunds_vs_Markets": ["fed", "rate hike", "rate cut", "fomc", "monetary policy"],
            "YieldCurve_Recession": ["yield curve", "inversion", "recession", "10y", "2y"],
            "VIX_Market_Stress": ["vix", "volatility", "fear", "stress"],
            "DXY_EM_Stocks": ["dollar", "dxy", "emerging", "em"],
            "Oil_Inflation_Earnings": ["oil", "crude", "energy", "wti", "brent"],
            "Gold_SafeHaven": ["gold", "safe haven", "precious"],
            "Unemployment_Stocks": ["unemployment", "jobs", "labor", "sahm"],
            "Earnings_StockPrice": ["earnings", "eps", "revenue", "beat", "miss"],
            "PCE_FedPolicy": ["pce", "fed", "core pce"],
            "China_GlobalTrade": ["china", "chinese", "shanghai", "hang seng"],
            "CreditSpreads_Risk": ["credit", "spread", "high yield", "default"],
            "TechEarnings_Nasdaq": ["tech", "magnificent", "ai", "cloud", "nasdaq"],
            "QE_QT_Liquidity": ["qe", "qt", "quantitative", "liquidity", "balance sheet"],
            "Sector_Rotation": ["sector", "rotation", "cycle", "defensive", "cyclical"],
            "Vietnam_Market": ["vietnam", "vn-index", "vnindex", "vn30"],
        }

        matches = []
        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    matches.append(topic)
                    break
        return matches

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        Returns a list of context snippets ordered by relevance.
        """
        self.load()
        contexts: List[Dict[str, Any]] = []

        # 1. Extract and retrieve stock data
        tickers = self._extract_tickers(query)
        for ticker in tickers[:5]:
            if ticker in self._stock_cache:
                contexts.append({
                    "type": "stock",
                    "relevance": 0.9,
                    "data": self._stock_cache[ticker],
                })
            if ticker in self._fundamentals_cache:
                contexts.append({
                    "type": "fundamentals",
                    "relevance": 0.85,
                    "data": self._fundamentals_cache[ticker],
                })

        # 2. Extract and retrieve macro data
        macro_keys = self._extract_macro_keywords(query)
        for key in macro_keys[:8]:
            if key in self._macro_cache:
                contexts.append({
                    "type": "macro",
                    "relevance": 0.85,
                    "data": self._macro_cache[key],
                })

        # 3. Retrieve correlation narratives
        topics = self._extract_narrative_topics(query)
        for topic in topics[:3]:
            for narr in self._narratives:
                if narr.get("topic") == topic:
                    contexts.append({
                        "type": "narrative",
                        "relevance": 0.8,
                        "data": narr,
                    })
                    break

        # 4. Search news (simple keyword matching)
        query_words = set(query.lower().split())
        relevant_news = []
        for art in self._news:
            title_words = set(art.get("title", "").lower().split())
            overlap = len(query_words & title_words)
            if overlap >= 2:
                relevant_news.append((overlap, art))
        relevant_news.sort(key=lambda x: x[0], reverse=True)
        for score, art in relevant_news[:3]:
            contexts.append({
                "type": "news",
                "relevance": min(0.7, score * 0.15),
                "data": art,
            })

        # Sort by relevance and limit
        contexts.sort(key=lambda x: x["relevance"], reverse=True)
        return contexts[:top_k]

    def build_context_string(self, query: str, max_tokens: int = 1500) -> str:
        """
        Build a context string from retrieved data, suitable for injection
        into an LLM prompt.
        """
        contexts = self.retrieve(query)
        if not contexts:
            return ""

        parts: List[str] = []
        current_length = 0

        for ctx in contexts:
            text = ctx["data"].get("text", "")
            if not text:
                # Construct text from data
                if ctx["type"] == "stock":
                    d = ctx["data"]
                    text = f"[STOCK DATA] {d.get('symbol', '')}: ${d.get('latest_price', 0):.2f}, returns: {d.get('returns', {})}"
                elif ctx["type"] == "macro":
                    d = ctx["data"]
                    text = f"[MACRO DATA] {d.get('indicator', '')}: {d.get('latest_value', 0):.4g} ({d.get('pct_change', 0):+.2f}% change)"
                elif ctx["type"] == "fundamentals":
                    text = ctx["data"].get("text", str(ctx["data"]))
                elif ctx["type"] == "news":
                    d = ctx["data"]
                    text = f"[NEWS] {d.get('title', '')} ({d.get('date', '')})"

            # Rough token estimate (4 chars ≈ 1 token)
            token_est = len(text) // 4
            if current_length + token_est > max_tokens:
                break
            parts.append(text)
            current_length += token_est

        return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  Singleton instance
# ══════════════════════════════════════════════════════════════

_rag_instance: Optional[FinancialRAG] = None

def get_rag() -> FinancialRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FinancialRAG()
    return _rag_instance
