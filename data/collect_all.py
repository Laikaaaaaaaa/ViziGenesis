#!/usr/bin/env python3
"""
ViziGenesis — Master Data Collection Script
============================================
Downloads as much free financial data as possible and organizes it into:

  data/
    macro/           # GDP, CPI, PCE, unemployment, money supply, trade balance, debt/GDP, ...
    markets/         # Indices, ETFs, commodities, FX, crypto, volatility, bonds
    stocks/          # Per-symbol OHLCV + technical indicators (US, VN, international)
    fundamentals/    # Revenue, EPS, P/E, margins, cash flow from Yahoo Finance
    news/            # Financial news from RSS feeds
    transcripts/     # Earnings call transcripts from SEC EDGAR
    central_bank/    # Fed speeches, FOMC minutes/statements

Run:  python data/collect_all.py [--section macro|markets|stocks|fundamentals|all]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

# ──────────────────────── Paths ────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MACRO_DIR     = DATA / "macro"
MARKETS_DIR   = DATA / "markets"
STOCKS_DIR    = DATA / "stocks"
FUND_DIR      = DATA / "fundamentals"
NEWS_DIR      = DATA / "news"
TRANSCRIPTS_DIR = DATA / "transcripts"
CENTRALBANK_DIR = DATA / "central_bank"

ALL_DIRS = [MACRO_DIR, MARKETS_DIR, STOCKS_DIR, FUND_DIR, NEWS_DIR, TRANSCRIPTS_DIR, CENTRALBANK_DIR]

# ──────────────────────── User-Agent ────────────────────────
UA = "ViziGenesis/3.0 (financial-research; +https://github.com/vizigenesis)"

# ══════════════════════════════════════════════════════════════
# 1. MACRO DATA — FRED + World Bank Data360
# ══════════════════════════════════════════════════════════════

# ---- FRED series (free public CSV, no API key needed) ----
FRED_SERIES: Dict[str, str] = {
    # Interest rates & Fed policy
    "DFF":              "fed_funds_rate",
    "DFEDTARU":         "fed_funds_upper_target",
    "DFEDTARL":         "fed_funds_lower_target",
    "EFFR":             "effective_fed_funds_rate",
    "IORB":             "interest_on_reserve_balances",
    "DISCOUNT":         "discount_rate",
    # Treasury yields
    "DGS1MO":           "treasury_yield_1m",
    "DGS3MO":           "treasury_yield_3m",
    "DGS6MO":           "treasury_yield_6m",
    "DGS1":             "treasury_yield_1y",
    "DGS2":             "treasury_yield_2y",
    "DGS3":             "treasury_yield_3y",
    "DGS5":             "treasury_yield_5y",
    "DGS7":             "treasury_yield_7y",
    "DGS10":            "treasury_yield_10y",
    "DGS20":            "treasury_yield_20y",
    "DGS30":            "treasury_yield_30y",
    "TB3MS":            "tbill_3m_secondary",
    # Yield spreads
    "T10Y2Y":           "term_spread_10y_2y",
    "T10Y3M":           "term_spread_10y_3m",
    "T10YFF":           "spread_10y_minus_fedfunds",
    # Inflation
    "CPIAUCSL":         "cpi_all_items",
    "CPILFESL":         "cpi_core_less_food_energy",
    "CPIUFDSL":         "cpi_food",
    "CPIENGSL":         "cpi_energy",
    "PCEPILFE":         "core_pce_price_index",
    "PCEPI":            "pce_price_index",
    "MEDCPIM158SFRBCLE":"median_cpi",
    "MICH":             "umich_inflation_expectations",
    "T5YIE":            "breakeven_inflation_5y",
    "T10YIE":           "breakeven_inflation_10y",
    # Employment
    "UNRATE":           "unemployment_rate",
    "PAYEMS":           "nonfarm_payrolls",
    "CIVPART":          "labor_force_participation",
    "JTSJOL":           "job_openings_jolts",
    "ICSA":             "initial_claims",
    "CCSA":             "continued_claims",
    "AWHAETP":          "avg_weekly_hours",
    "CES0500000003":    "avg_hourly_earnings",
    # GDP & output
    "GDP":              "gdp_nominal",
    "GDPC1":            "gdp_real",
    "A191RL1Q225SBEA":  "gdp_real_growth_annualized",
    "GDPPOT":           "gdp_potential",
    "INDPRO":           "industrial_production",
    "TCU":              "capacity_utilization",
    # Consumer & housing
    "UMCSENT":          "umich_consumer_sentiment",
    "RSXFS":            "retail_sales_ex_food_services",
    "RRSFS":            "real_retail_sales",
    "HOUST":            "housing_starts",
    "PERMIT":           "building_permits",
    "CSUSHPISA":        "case_shiller_home_price_index",
    "MORTGAGE30US":     "mortgage_rate_30y",
    "MORTGAGE15US":     "mortgage_rate_15y",
    # Money & credit
    "M1SL":             "m1_money_stock",
    "M2SL":             "m2_money_stock",
    "WALCL":            "fed_balance_sheet_total_assets",
    "TOTRESNS":         "total_reserves",
    "RRPONTSYD":        "reverse_repo_overnight",
    "WTREGEN":          "treasury_general_account",
    "TOTBKCR":          "bank_credit_total",
    "REVOLSL":          "consumer_credit_revolving",
    "NONREVSL":         "consumer_credit_nonrevolving",
    # Trade & international
    "BOPGSTB":          "trade_balance",
    "TWEXBGSMTH":       "trade_weighted_usd_broad",
    # Corporate & credit spreads
    "BAA10Y":           "baa_spread_over_10y",
    "AAA10Y":           "aaa_spread_over_10y",
    "BAMLH0A0HYM2":    "hy_oas_spread",
    "BAMLC0A4CBBB":     "bbb_oas_spread",
    "BAMLHE00EHYIEY":   "hy_effective_yield",
    # Equity market
    "SP500":            "sp500_index",
    "NASDAQCOM":        "nasdaq_composite",
    "WILSHIRE5000PRFC": "wilshire_5000",
    "VIXCLS":           "vix_close",
    # Commodities (FRED has these)
    "DCOILWTICO":       "wti_crude_spot",
    "DCOILBRENTEU":     "brent_crude_spot",
    "GOLDAMGBD228NLBM": "gold_london_fix_pm",
    # FX (FRED daily)
    "DEXUSEU":          "usd_eur",
    "DEXJPUS":          "usd_jpy",
    "DEXUSUK":          "usd_gbp",
    "DEXCHUS":          "usd_cny",
    "DEXCAUS":          "usd_cad",
    "DEXSZUS":          "usd_chf",
    "DEXMXUS":          "usd_mxn",
    "DEXBZUS":          "usd_brl",
    "DEXINUS":          "usd_inr",
    "DEXKOUS":          "usd_krw",
    # PMI / leading indicators
    "MANEMP":           "manufacturing_employment",
    "AMTMTI":           "manufacturers_total_inventories",
    "NEWORDER":         "new_orders_nondefense",
    "DGORDER":          "durable_goods_orders",
    "USSLIND":          "leading_economic_index",
}

# ---- World Bank Data360 indicators ----
DATA360_INDICATORS: Dict[str, Dict[str, str]] = {
    "WB_GDP_GROWTH":      {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "WB_GDP_PER_CAPITA":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_PCAP_CD", "REF_AREA": "USA", "FREQ": "A"},
    "WB_CPI":             {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL", "REF_AREA": "USA", "FREQ": "A"},
    "WB_INFLATION":       {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "USA", "FREQ": "A"},
    "WB_UNEMPLOYMENT":    {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_SL_UEM_TOTL_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_LENDING_RATE":    {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FR_INR_LEND", "REF_AREA": "USA", "FREQ": "A"},
    "WB_M2_GDP":          {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FM_LBL_BMNY_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_TRADE_GDP":       {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NE_TRD_GNFS_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_FDI_GDP":         {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BX_KLT_DINV_WD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_GOV_DEBT_GDP":    {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_GC_DOD_TOTL_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_CURR_ACCOUNT":    {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_BN_CAB_XOKA_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_STOCKS_TRADED":   {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_TRAD_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    "WB_MCAP_GDP":        {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_CM_MKT_LCAP_GD_ZS", "REF_AREA": "USA", "FREQ": "A"},
    # Add more countries for cross-country analysis
    "WB_GDP_GROWTH_CHN":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "CHN", "FREQ": "A"},
    "WB_GDP_GROWTH_JPN":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "JPN", "FREQ": "A"},
    "WB_GDP_GROWTH_DEU":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "DEU", "FREQ": "A"},
    "WB_GDP_GROWTH_GBR":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "GBR", "FREQ": "A"},
    "WB_GDP_GROWTH_VNM":  {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_NY_GDP_MKTP_KD_ZG", "REF_AREA": "VNM", "FREQ": "A"},
    "WB_CPI_CHN":         {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "CHN", "FREQ": "A"},
    "WB_CPI_JPN":         {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "JPN", "FREQ": "A"},
    "WB_CPI_VNM":         {"DATABASE_ID": "WB_WDI", "INDICATOR": "WB_WDI_FP_CPI_TOTL_ZG", "REF_AREA": "VNM", "FREQ": "A"},
}
DATA360_URL = "https://data360api.worldbank.org/data360/data"


# ══════════════════════════════════════════════════════════════
# 2. MARKET DATA — Yahoo Finance tickers for indices, ETFs,
#    commodities, FX, crypto, bonds, volatility
# ══════════════════════════════════════════════════════════════

MARKET_TICKERS: Dict[str, str] = {
    # ── Major US indices ──
    "SP500":       "^GSPC",
    "NASDAQ":      "^IXIC",
    "NASDAQ100":   "^NDX",
    "DOW":         "^DJI",
    "RUSSELL2000":  "^RUT",
    "RUSSELL1000":  "^RUI",
    "SP400_MID":   "^MID",
    "SP600_SMALL": "^SML",
    "NYSE_COMP":   "^NYA",
    "WILSHIRE5000":"^W5000",
    # ── International indices ──
    "FTSE100":     "^FTSE",
    "DAX":         "^GDAXI",
    "CAC40":       "^FCHI",
    "NIKKEI225":   "^N225",
    "HANG_SENG":   "^HSI",
    "SHANGHAI":    "000001.SS",
    "KOSPI":       "^KS11",
    "SENSEX":      "^BSESN",
    "ASX200":      "^AXJO",
    "TSX":         "^GSPTSE",
    "BOVESPA":     "^BVSP",
    "EURO_STOXX50":"^STOXX50E",
    "VNINDEX":     "^VNINDEX",
    # ── Volatility ──
    "VIX":         "^VIX",
    "VXN":         "^VXN",
    "VVIX":        "^VVIX",
    "MOVE":        "^MOVE",  # Bond vol
    # ── Treasury yields (Yahoo) ──
    "YIELD_13W":   "^IRX",
    "YIELD_5Y":    "^FVX",
    "YIELD_10Y":   "^TNX",
    "YIELD_30Y":   "^TYX",
    # ── Bond ETFs ──
    "TLT": "TLT",  "IEF": "IEF",  "SHY": "SHY",
    "HYG": "HYG",  "LQD": "LQD",  "TIP": "TIP",
    "AGG": "AGG",  "BND": "BND",  "BNDX": "BNDX",
    "EMB": "EMB",  "JNK": "JNK",  "MUB": "MUB",
    # ── Sector ETFs (all 11 GICS sectors + semis + clean energy) ──
    "XLK": "XLK",  "XLF": "XLF",  "XLE": "XLE",
    "XLV": "XLV",  "XLI": "XLI",  "XLP": "XLP",
    "XLY": "XLY",  "XLB": "XLB",  "XLU": "XLU",
    "XLRE": "XLRE", "XLC": "XLC",
    "SOXX": "SOXX", "SMH": "SMH",
    "ARKK": "ARKK", "ARKW": "ARKW",
    "ICLN": "ICLN", "TAN": "TAN",  "LIT": "LIT",
    # ── Thematic / Factor ETFs ──
    "QQQ": "QQQ",  "SPY": "SPY",  "IWM": "IWM",
    "DIA": "DIA",  "VOO": "VOO",  "VTI": "VTI",
    "MTUM": "MTUM","VLUE":"VLUE","QUAL":"QUAL",
    "SIZE": "SIZE","USMV":"USMV",
    # ── Commodities ──
    "GOLD":      "GC=F",
    "SILVER":    "SI=F",
    "PLATINUM":  "PL=F",
    "PALLADIUM": "PA=F",
    "OIL_WTI":   "CL=F",
    "OIL_BRENT": "BZ=F",
    "NATGAS":    "NG=F",
    "COPPER":    "HG=F",
    "CORN":      "ZC=F",
    "WHEAT":     "ZW=F",
    "SOYBEANS":  "ZS=F",
    "COTTON":    "CT=F",
    "SUGAR":     "SB=F",
    "COFFEE":    "KC=F",
    "LUMBER":    "LBS=F",
    # ── Commodity ETFs ──
    "GLD":  "GLD",  "SLV":  "SLV",  "USO":  "USO",
    "UNG":  "UNG",  "DBA":  "DBA",  "DBC":  "DBC",
    "PDBC": "PDBC",
    # ── FX Pairs ──
    "EURUSD":  "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",
    "USDJPY":  "USDJPY=X",
    "USDCHF":  "USDCHF=X",
    "USDCAD":  "USDCAD=X",
    "AUDUSD":  "AUDUSD=X",
    "NZDUSD":  "NZDUSD=X",
    "USDCNY":  "USDCNY=X",
    "USDHKD":  "USDHKD=X",
    "USDKRW":  "USDKRW=X",
    "USDINR":  "USDINR=X",
    "USDBRL":  "USDBRL=X",
    "USDMXN":  "USDMXN=X",
    "USDTRY":  "USDTRY=X",
    "USDZAR":  "USDZAR=X",
    "USDTWD":  "USDTWD=X",
    "USDSGD":  "USDSGD=X",
    "USDVND":  "USDVND=X",
    "DXY":     "DX-Y.NYB",
    # ── Crypto ──
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "BNB":  "BNB-USD",
    "XRP":  "XRP-USD",
    "ADA":  "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT":  "DOT-USD",
    "LINK": "LINK-USD",
    # ── International ETFs ──
    "EEM": "EEM",  "EFA": "EFA",  "FXI": "FXI",
    "EWJ": "EWJ",  "EWZ": "EWZ",  "EWG": "EWG",
    "EWY": "EWY",  "EWT": "EWT",  "EWA": "EWA",
    "INDA": "INDA","VWO": "VWO",  "IEMG": "IEMG",
    "MCHI": "MCHI","KWEB": "KWEB",
}

# ══════════════════════════════════════════════════════════════
# 3. STOCK TICKERS — US, Vietnam, International, INDEX
#    constituents
# ══════════════════════════════════════════════════════════════

STOCK_TICKERS: List[str] = [
    # ── US Mega-cap / Tech (30) ──
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "INTC", "AMD", "QCOM", "TXN",
    "AMAT", "LRCX", "KLAC", "MU", "ANET", "PANW", "NOW", "SNOW",
    "DDOG", "PLTR", "SHOP", "UBER", "ABNB", "NFLX",
    # ── More tech / software (20) ──
    "SPOT", "SQ", "COIN", "RBLX", "TWLO", "ZM", "CRWD", "NET",
    "MDB", "ZS", "DOCU", "WDAY", "TEAM", "HUBS", "VEEV", "TTD",
    "BILL", "OKTA", "DASH", "LYFT",
    # ── Finance (20) ──
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP",
    "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC", "BK", "STT",
    "ICE", "CME",
    # ── Healthcare / Pharma / Biotech (25) ──
    "UNH", "JNJ", "PFE", "MRK", "LLY", "ABBV", "TMO", "DHR",
    "ABT", "ISRG", "GILD", "BMY", "AMGN", "VRTX", "REGN",
    "BIIB", "ILMN", "MRNA", "DXCM", "EW", "ZTS", "IDXX",
    "SYK", "BSX", "MDT",
    # ── Consumer discretionary / staples (25) ──
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS",
    "CMCSA", "PEP", "KO", "PG", "PM", "MO", "EL", "TGT",
    "ROST", "TJX", "LULU", "BKNG", "MAR", "HLT", "DKNG",
    "YUM", "CMG",
    # ── Industrials / Defense / Aerospace (20) ──
    "BA", "CAT", "DE", "UNP", "HON", "GE", "RTX", "LMT",
    "NOC", "GD", "ETN", "PH", "WM", "RSG", "CSX", "NSC",
    "FAST", "EMR", "IR", "CARR",
    # ── Energy (15) ──
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO",
    "PSX", "HES", "DVN", "FANG", "HAL", "BKR", "CTRA",
    # ── Utilities / REITs / Telecom (15) ──
    "NEE", "DUK", "SO", "AEP", "EXC", "AMT", "PLD", "EQIX",
    "CCI", "O", "T", "VZ", "TMUS", "SPG", "PSA",
    # ── EV / Clean energy ──
    "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR",
    # ── Vietnam ──
    "VIC.VN", "VNM.VN", "HPG.VN", "FPT.VN", "VCB.VN",
    "SSI.VN", "MBB.VN", "ACB.VN", "MWG.VN", "VHM.VN",
    "TCB.VN", "CTG.VN", "BID.VN", "MSN.VN", "VRE.VN",
    "GVR.VN", "PLX.VN", "POW.VN", "REE.VN", "PNJ.VN",
    # ── Other Asia / EM ──
    "TSM",  "BABA", "JD", "PDD", "BIDU", "SE",
    "GRAB", "MELI", "NU", "RELIANCE.NS", "TCS.NS",
    "005930.KS", "000660.KS",  # Samsung, SK Hynix
    "9984.T", "6758.T", "7203.T",  # SoftBank, Sony, Toyota
]

# ── Index ETFs that represent index30/50/100-type products ──
INDEX_PRODUCTS: Dict[str, str] = {
    # US indices
    "index_sp500":      "SPY",
    "index_sp100":      "OEF",
    "index_nasdaq100":  "QQQ",
    "index_dow30":      "DIA",
    "index_russell2000":"IWM",
    "index_russell1000":"IWB",
    "index_sp_midcap":  "MDY",
    "index_sp_smallcap":"IJR",
    # Vietnam
    "index_vn30":       "E1VFVN30.VN",
    "index_vndia":      "FUEVFVND.VN",
    # International
    "index_ftse100":    "ISF.L",
    "index_dax":        "EXS1.DE",
    "index_nikkei225":  "1321.T",
    "index_hang_seng":  "2800.HK",
    "index_csi300":     "510300.SS",
    "index_kospi200":   "069500.KS",
    "index_sensex":     "0P0001BSEG.BO",
    "index_asx200":     "IOZ.AX",
    "index_msci_em":    "EEM",
    "index_msci_world": "URTH",
    "index_msci_eafe":  "EFA",
}

# ══════════════════════════════════════════════════════════════
# SEC EDGAR Configuration (free, no API key, rate limit = 10/sec)
# ══════════════════════════════════════════════════════════════
SEC_HEADERS = {"User-Agent": "ViziGenesis research@vizigenesis.com", "Accept-Encoding": "gzip, deflate"}

# CIK lookup for major companies (SEC uses CIK numbers)
EDGAR_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "GS", "V", "MA", "UNH", "JNJ", "PFE",
    "XOM", "CVX", "WMT", "COST", "HD", "DIS", "NFLX",
    "BA", "CAT", "GE", "AMD", "INTC", "CRM", "AVGO", "ORCL",
]

# ══════════════════════════════════════════════════════════════
# RSS Feeds for financial news
# ══════════════════════════════════════════════════════════════
NEWS_RSS_FEEDS: Dict[str, str] = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets":  "https://feeds.reuters.com/reuters/marketsNews",
    "wsj_markets":      "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc_economy":     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "cnbc_finance":     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "yahoo_finance":    "https://finance.yahoo.com/news/rssindex",
    "ft_markets":       "https://www.ft.com/markets?format=rss",
    "fed_speeches":     "https://www.federalreserve.gov/feeds/speeches.xml",
    "fed_press":        "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "ecb_press":        "https://www.ecb.europa.eu/rss/press.html",
}


# ══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    for p in ALL_DIRS:
        p.mkdir(parents=True, exist_ok=True)
    # Sub-directories
    for sub in ["fred", "data360", "data360_intl"]:
        (MACRO_DIR / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["indices", "etfs", "commodities", "fx", "crypto", "bonds", "volatility", "sectors"]:
        (MARKETS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["rss", "articles"]:
        (NEWS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["fed_speeches", "fomc_minutes", "fomc_statements"]:
        (CENTRALBANK_DIR / sub).mkdir(parents=True, exist_ok=True)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def _safe_request(url: str, timeout: int = 25, retries: int = 2, headers: Optional[Dict[str, str]] = None) -> Optional[bytes]:
    """Perform an HTTP GET with retries. Returns raw bytes or None."""
    import gzip as _gzip
    for attempt in range(retries):
        try:
            hdrs = headers or {"User-Agent": UA}
            req = Request(url, headers=hdrs)
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Handle gzip-encoded responses
                if raw[:2] == b'\x1f\x8b':
                    raw = _gzip.decompress(raw)
                return raw
        except (HTTPError, URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(1 + attempt)
    return None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = _safe_div(avg_up, avg_down)
    return 100 - (100 / (1 + rs))


def _safe_div(num, denom):
    """Divide with NaN where denominator is 0, keeping float dtype."""
    import numpy as np
    d = denom.astype(float)
    d = d.replace(0.0, np.nan)
    return num.astype(float) / d


def build_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute 30+ technical indicators from OHLCV data."""
    req = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in ohlcv.columns for c in req):
        return pd.DataFrame(index=ohlcv.index)

    c = pd.to_numeric(ohlcv["Close"], errors="coerce")
    h = pd.to_numeric(ohlcv["High"], errors="coerce")
    lo = pd.to_numeric(ohlcv["Low"], errors="coerce")
    op = pd.to_numeric(ohlcv["Open"], errors="coerce")
    vol = pd.to_numeric(ohlcv["Volume"], errors="coerce").fillna(0)

    ind = pd.DataFrame(index=ohlcv.index)

    # Returns
    ind["Return_1d"]  = c.pct_change(fill_method=None) * 100
    ind["Return_5d"]  = c.pct_change(5, fill_method=None) * 100
    ind["Return_10d"] = c.pct_change(10, fill_method=None) * 100
    ind["Return_20d"] = c.pct_change(20, fill_method=None) * 100
    ind["Return_60d"] = c.pct_change(60, fill_method=None) * 100

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        ind[f"SMA_{w}"] = c.rolling(w, min_periods=max(5, w // 4)).mean()
    for w in [9, 12, 20, 26, 50]:
        ind[f"EMA_{w}"] = c.ewm(span=w, adjust=False).mean()

    # RSI
    ind["RSI_14"] = _rsi(c, 14)
    ind["RSI_7"]  = _rsi(c, 7)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ind["MACD"] = ema12 - ema26
    ind["MACD_Signal"] = ind["MACD"].ewm(span=9, adjust=False).mean()
    ind["MACD_Hist"] = ind["MACD"] - ind["MACD_Signal"]

    # Bollinger Bands
    for w in [20]:
        bb_mid = c.rolling(w, min_periods=5).mean()
        bb_std = c.rolling(w, min_periods=5).std(ddof=0)
        ind[f"BB_Upper_{w}"] = bb_mid + 2 * bb_std
        ind[f"BB_Lower_{w}"] = bb_mid - 2 * bb_std
        ind[f"BB_Width_{w}"] = _safe_div(ind[f"BB_Upper_{w}"] - ind[f"BB_Lower_{w}"], bb_mid)
        ind[f"BB_PctB_{w}"] = _safe_div(c - ind[f"BB_Lower_{w}"], ind[f"BB_Upper_{w}"] - ind[f"BB_Lower_{w}"])

    # ATR
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - lo).abs(),
        (h - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)
    ind["ATR_14"] = tr.rolling(14, min_periods=5).mean()
    ind["ATR_7"]  = tr.rolling(7, min_periods=3).mean()

    # OBV
    direction = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    ind["OBV"] = (direction * vol).cumsum()

    # ROC / Momentum
    ind["ROC_10"] = c.pct_change(10, fill_method=None) * 100
    ind["ROC_20"] = c.pct_change(20, fill_method=None) * 100
    ind["Momentum_10"] = c - c.shift(10)
    ind["Momentum_20"] = c - c.shift(20)

    # Volatility
    ret = c.pct_change(fill_method=None)
    ind["HistVol_10"] = ret.rolling(10, min_periods=5).std(ddof=0) * (252 ** 0.5) * 100
    ind["HistVol_20"] = ret.rolling(20, min_periods=5).std(ddof=0) * (252 ** 0.5) * 100
    ind["HistVol_60"] = ret.rolling(60, min_periods=10).std(ddof=0) * (252 ** 0.5) * 100

    # VWAP
    typical = (h + lo + c) / 3
    ind["VWAP"] = _safe_div((typical * vol).cumsum(), vol.cumsum())

    # Stochastic oscillator
    for w in [14]:
        l14 = lo.rolling(w, min_periods=5).min()
        h14 = h.rolling(w, min_periods=5).max()
        ind[f"Stoch_K_{w}"] = _safe_div(c - l14, h14 - l14) * 100
        ind[f"Stoch_D_{w}"] = ind[f"Stoch_K_{w}"].rolling(3).mean()

    # Williams %R
    h14 = h.rolling(14, min_periods=5).max()
    l14 = lo.rolling(14, min_periods=5).min()
    ind["Williams_R"] = _safe_div(h14 - c, h14 - l14) * -100

    # CCI (Commodity Channel Index)
    typical = (h + lo + c) / 3
    sma_tp = typical.rolling(20, min_periods=5).mean()
    mad_tp = typical.rolling(20, min_periods=5).apply(lambda x: __import__('numpy').abs(x - x.mean()).mean(), raw=True)
    ind["CCI_20"] = _safe_div(typical - sma_tp, 0.015 * mad_tp)

    # AD Line (Accumulation/Distribution)
    mfm = _safe_div((c - lo) - (h - c), h - lo)
    ind["AD_Line"] = (mfm.fillna(0) * vol).cumsum()

    # Money Flow Index
    typical = (h + lo + c) / 3
    mf = typical * vol
    pos_mf = mf.where(typical > typical.shift(1), 0)
    neg_mf = mf.where(typical < typical.shift(1), 0)
    pmf14 = pos_mf.rolling(14, min_periods=5).sum()
    nmf14 = neg_mf.rolling(14, min_periods=5).sum()
    mfr = _safe_div(pmf14, nmf14)
    ind["MFI_14"] = 100 - (100 / (1 + mfr))

    # Average Directional Index (ADX)
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-lo.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr14 = ind["ATR_14"]
    plus_di = 100 * _safe_div(plus_dm.ewm(span=14, adjust=False).mean(), atr14)
    minus_di = 100 * _safe_div(minus_dm.ewm(span=14, adjust=False).mean(), atr14)
    dx = _safe_div((plus_di - minus_di).abs(), plus_di + minus_di) * 100
    ind["ADX_14"] = dx.ewm(span=14, adjust=False).mean()

    # Ichimoku cloud components
    ind["Ichimoku_Tenkan"]  = (h.rolling(9, min_periods=5).max() + lo.rolling(9, min_periods=5).min()) / 2
    ind["Ichimoku_Kijun"]   = (h.rolling(26, min_periods=10).max() + lo.rolling(26, min_periods=10).min()) / 2
    ind["Ichimoku_SenkouA"] = ((ind["Ichimoku_Tenkan"] + ind["Ichimoku_Kijun"]) / 2).shift(26)
    ind["Ichimoku_SenkouB"] = ((h.rolling(52, min_periods=20).max() + lo.rolling(52, min_periods=20).min()) / 2).shift(26)

    # Parabolic SAR approximation (simplified)
    ind["Price_vs_SMA200"] = (_safe_div(c, ind.get("SMA_200", c)) - 1) * 100

    # Volume indicators
    ind["Volume_SMA_20"] = vol.rolling(20, min_periods=5).mean()
    ind["Volume_Ratio"]  = _safe_div(vol, vol.rolling(20, min_periods=5).mean())
    ind["Force_Index"]   = c.diff() * vol

    # Gap
    ind["Gap_Pct"] = _safe_div(op - c.shift(1), c.shift(1)) * 100

    return ind


# ══════════════════════════════════════════════════════════════
#  DOWNLOAD FUNCTIONS
# ══════════════════════════════════════════════════════════════

def download_yahoo_symbol(symbol: str, period: str = "max", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            df = _normalize_index(df)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.8 + attempt * 0.6)
    return pd.DataFrame()


# ──────────────────── MACRO ────────────────────

def download_fred_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        if len(df.columns) >= 2:
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            return df
    except Exception:
        pass
    return pd.DataFrame()


def collect_macro() -> Dict[str, Any]:
    """Download all macro data (FRED + Data360)."""
    print("\n=== MACRO DATA ===")
    stats: Dict[str, Any] = {"fred": {}, "data360": {}}

    # FRED
    print(f"  Downloading {len(FRED_SERIES)} FRED series...")
    fred_dir = MACRO_DIR / "fred"
    for i, (sid, name) in enumerate(FRED_SERIES.items(), 1):
        df = download_fred_csv(sid)
        if df.empty:
            stats["fred"][name] = 0
            continue
        df.to_csv(fred_dir / f"{name}.csv")
        stats["fred"][name] = len(df)
        if i % 20 == 0:
            print(f"    FRED: {i}/{len(FRED_SERIES)} done ...")

    ok = sum(1 for v in stats["fred"].values() if v > 0)
    print(f"  FRED: {ok}/{len(FRED_SERIES)} series downloaded")

    # Data360
    print(f"  Downloading {len(DATA360_INDICATORS)} Data360 indicators...")
    for name, cfg in DATA360_INDICATORS.items():
        try:
            df = _fetch_data360(cfg)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            stats["data360"][name] = 0
            continue
        subdir = "data360_intl" if name.endswith(("_CHN", "_JPN", "_DEU", "_GBR", "_VNM")) else "data360"
        df.to_csv(MACRO_DIR / subdir / f"{name}.csv")
        stats["data360"][name] = len(df)

    ok = sum(1 for v in stats["data360"].values() if v > 0)
    print(f"  Data360: {ok}/{len(DATA360_INDICATORS)} indicators downloaded")

    return stats


def _fetch_data360(cfg: Dict[str, str], from_year: str = "1960") -> pd.DataFrame:
    params = {
        "DATABASE_ID": cfg.get("DATABASE_ID", ""),
        "INDICATOR": cfg.get("INDICATOR", ""),
        "timePeriodFrom": from_year,
        "format": "json",
        "top": 1000,
        "skip": 0,
    }
    if cfg.get("REF_AREA"):
        params["REF_AREA"] = cfg["REF_AREA"]
    if cfg.get("FREQ"):
        params["FREQ"] = cfg["FREQ"]

    dates, vals = [], []
    expected = None

    while True:
        raw = _safe_request(f"{DATA360_URL}?{urlencode(params)}", timeout=20)
        if raw is None:
            break
        payload = json.loads(raw.decode("utf-8"))
        rows = payload.get("value", []) if isinstance(payload, dict) else []
        if expected is None and isinstance(payload, dict):
            try:
                expected = int(payload.get("count", 0))
            except Exception:
                expected = 0
        if not rows:
            break
        for r in rows:
            t = pd.to_datetime(str(r.get("TIME_PERIOD", "")).strip(), errors="coerce")
            try:
                v = float(r.get("OBS_VALUE"))
            except Exception:
                continue
            if pd.isna(t):
                continue
            dates.append(t)
            vals.append(v)
        params["skip"] = int(params.get("skip", 0)) + int(params.get("top", 1000))
        if len(rows) < int(params.get("top", 1000)):
            break
        if expected is not None and int(params["skip"]) >= expected:
            break

    if not dates:
        return pd.DataFrame()
    df = pd.DataFrame({"value": vals}, index=pd.DatetimeIndex(dates)).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ──────────────────── MARKETS ────────────────────

def _categorize_market_ticker(name: str) -> str:
    """Determine subdirectory for a market ticker."""
    lname = name.lower()
    if any(x in lname for x in ["sp500", "nasdaq", "dow", "russell", "nyse", "wilshire",
                                  "ftse", "dax", "cac", "nikkei", "hang", "shanghai",
                                  "kospi", "sensex", "asx", "tsx", "bovespa", "euro_stoxx",
                                  "vnindex"]):
        return "indices"
    if any(x in lname for x in ["vix", "vxn", "vvix", "move"]):
        return "volatility"
    if any(x in lname for x in ["yield", "tlt", "ief", "shy", "hyg", "lqd", "tip",
                                  "agg", "bnd", "emb", "jnk", "mub"]):
        return "bonds"
    if any(x in lname for x in ["gold", "silver", "platinum", "palladium", "oil",
                                  "natgas", "copper", "corn", "wheat", "soy", "cotton",
                                  "sugar", "coffee", "lumber", "gld", "slv", "uso",
                                  "ung", "dba", "dbc", "pdbc"]):
        return "commodities"
    if any(x in lname for x in ["usd", "eur", "gbp", "jpy", "chf", "cad", "aud",
                                  "nzd", "cny", "hkd", "krw", "inr", "brl", "mxn",
                                  "try", "zar", "twd", "sgd", "vnd", "dxy"]):
        return "fx"
    if any(x in lname for x in ["btc", "eth", "sol", "bnb", "xrp", "ada",
                                  "doge", "avax", "dot", "link"]):
        return "crypto"
    if any(x in lname for x in ["xl", "soxx", "smh", "ark", "icln", "tan", "lit"]):
        return "sectors"
    if any(x in lname for x in ["eem", "efa", "fxi", "ewj", "ewz", "ewg", "ewy",
                                  "ewt", "ewa", "inda", "vwo", "iemg", "mchi", "kweb",
                                  "qqq", "spy", "iwm", "dia", "voo", "vti",
                                  "mtum", "vlue", "qual", "size", "usmv"]):
        return "etfs"
    return "etfs"


def collect_markets() -> Dict[str, int]:
    """Download all market instruments."""
    print("\n=== MARKET DATA ===")
    rows: Dict[str, int] = {}
    total = len(MARKET_TICKERS)
    for i, (name, ticker) in enumerate(MARKET_TICKERS.items(), 1):
        df = download_yahoo_symbol(ticker, period="max")
        if df.empty:
            rows[name] = 0
            continue
        subdir = _categorize_market_ticker(name)
        out_path = MARKETS_DIR / subdir / f"{name}.csv"
        df.to_csv(out_path)
        rows[name] = len(df)
        if i % 20 == 0:
            print(f"  Markets: {i}/{total} done ...")
            time.sleep(0.3)

    ok = sum(1 for v in rows.values() if v > 0)
    print(f"  Markets: {ok}/{total} instruments downloaded")
    return rows


# ──────────────────── STOCKS (with tech indicators) ────────────────────

def collect_stocks() -> Dict[str, int]:
    """Download stock OHLCV + technical indicators."""
    print("\n=== STOCK DATA ===")
    all_symbols = list(set(STOCK_TICKERS))
    rows: Dict[str, int] = {}
    total = len(all_symbols)

    for i, sym in enumerate(all_symbols, 1):
        df = download_yahoo_symbol(sym, period="max")
        if df.empty:
            rows[sym] = 0
            continue

        stock_dir = STOCKS_DIR / sym
        stock_dir.mkdir(parents=True, exist_ok=True)

        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        ohlcv = df.copy()
        for col in req_cols:
            if col not in ohlcv.columns:
                ohlcv[col] = pd.NA
        ohlcv = ohlcv[req_cols].dropna(how="all")

        indicators = build_technical_indicators(ohlcv)
        features = pd.concat([ohlcv, indicators], axis=1)

        ohlcv.to_csv(stock_dir / "ohlcv.csv")
        indicators.to_csv(stock_dir / "technical_indicators.csv")
        features.to_csv(stock_dir / "features.csv")

        rows[sym] = len(df)
        if i % 20 == 0:
            print(f"  Stocks: {i}/{total} done ...")
            time.sleep(0.3)

    ok = sum(1 for v in rows.values() if v > 0)
    print(f"  Stocks: {ok}/{total} symbols downloaded")
    return rows


def collect_indices() -> Dict[str, int]:
    """Download index ETFs with full history → data/stocks/index_xxx/"""
    print("\n=== INDEX PRODUCTS ===")
    rows: Dict[str, int] = {}
    for name, ticker in INDEX_PRODUCTS.items():
        df = download_yahoo_symbol(ticker, period="max")
        if df.empty:
            rows[name] = 0
            continue

        idx_dir = STOCKS_DIR / name
        idx_dir.mkdir(parents=True, exist_ok=True)

        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all") if "Volume" in df.columns else df[["Open", "High", "Low", "Close"]]
        indicators = build_technical_indicators(ohlcv) if "Volume" in ohlcv.columns else pd.DataFrame(index=ohlcv.index)
        features = pd.concat([ohlcv, indicators], axis=1)

        ohlcv.to_csv(idx_dir / "ohlcv.csv")
        if not indicators.empty:
            indicators.to_csv(idx_dir / "technical_indicators.csv")
        features.to_csv(idx_dir / "features.csv")

        rows[name] = len(df)

    ok = sum(1 for v in rows.values() if v > 0)
    print(f"  Index products: {ok}/{len(INDEX_PRODUCTS)} downloaded")
    return rows


# ──────────────────── FUNDAMENTALS ────────────────────

def collect_fundamentals() -> Dict[str, int]:
    """
    Download corporate fundamentals from Yahoo Finance.
    For each ticker: financials, balance sheet, cash flow, key stats.
    """
    print("\n=== FUNDAMENTALS ===")
    stats: Dict[str, int] = {}
    tickers = list(set(STOCK_TICKERS))  # all stocks
    total = len(tickers)

    for i, sym in enumerate(tickers, 1):
        try:
            t = yf.Ticker(sym)
            fund_dir = FUND_DIR / sym
            fund_dir.mkdir(parents=True, exist_ok=True)
            saved = 0

            # Income statement
            for attr, fname in [
                ("financials", "income_annual.csv"),
                ("quarterly_financials", "income_quarterly.csv"),
                ("balance_sheet", "balance_sheet_annual.csv"),
                ("quarterly_balance_sheet", "balance_sheet_quarterly.csv"),
                ("cashflow", "cashflow_annual.csv"),
                ("quarterly_cashflow", "cashflow_quarterly.csv"),
            ]:
                try:
                    df = getattr(t, attr)
                    if df is not None and not df.empty:
                        df.to_csv(fund_dir / fname)
                        saved += 1
                except Exception:
                    pass

            # Key stats
            try:
                info = t.info
                if info:
                    # Extract relevant financial metrics
                    metrics = {}
                    keys = [
                        "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
                        "pegRatio", "priceToSalesTrailing12Months", "priceToBook",
                        "enterpriseToRevenue", "enterpriseToEbitda",
                        "profitMargins", "operatingMargins", "grossMargins",
                        "returnOnAssets", "returnOnEquity",
                        "revenue", "revenueGrowth", "revenuePerShare",
                        "totalDebt", "debtToEquity", "totalCash", "totalCashPerShare",
                        "earningsGrowth", "earningsQuarterlyGrowth",
                        "trailingEps", "forwardEps",
                        "bookValue", "dividendYield", "dividendRate",
                        "payoutRatio", "beta", "52WeekChange",
                        "shortRatio", "shortPercentOfFloat",
                        "freeCashflow", "operatingCashflow",
                        "currentRatio", "quickRatio",
                        "sector", "industry", "fullTimeEmployees",
                    ]
                    for k in keys:
                        if k in info:
                            metrics[k] = info[k]
                    if metrics:
                        with open(fund_dir / "key_stats.json", "w", encoding="utf-8") as f:
                            json.dump(metrics, f, indent=2, default=str)
                        saved += 1
            except Exception:
                pass

            # Earnings dates & EPS
            try:
                earn = t.earnings_dates
                if earn is not None and not earn.empty:
                    earn = _normalize_index(earn)
                    earn.to_csv(fund_dir / "earnings_dates.csv")
                    saved += 1
            except Exception:
                pass

            # Recommendations
            try:
                rec = t.recommendations
                if rec is not None and not rec.empty:
                    rec.to_csv(fund_dir / "recommendations.csv")
                    saved += 1
            except Exception:
                pass

            # Institutional holders
            try:
                inst = t.institutional_holders
                if inst is not None and not inst.empty:
                    inst.to_csv(fund_dir / "institutional_holders.csv")
                    saved += 1
            except Exception:
                pass

            stats[sym] = saved

        except Exception:
            stats[sym] = 0

        if i % 20 == 0:
            print(f"  Fundamentals: {i}/{total} done ...")
            time.sleep(0.5)

    ok = sum(1 for v in stats.values() if v > 0)
    print(f"  Fundamentals: {ok}/{total} companies with data")
    return stats


# ──────────────────── SEC EDGAR: Earnings Transcripts ────────────────────

def _sec_get_cik(ticker: str) -> Optional[str]:
    """Look up SEC CIK for a ticker."""
    try:
        raw = _safe_request("https://www.sec.gov/files/company_tickers.json", timeout=15, headers=SEC_HEADERS)
        if raw:
            data = json.loads(raw.decode("utf-8"))
            for entry in data.values():
                if str(entry.get("ticker", "")).upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def collect_sec_filings() -> Dict[str, int]:
    """
    Download recent SEC filings (10-K, 10-Q, 8-K) from EDGAR full-text search.
    These contain earnings data, management discussion, risk factors.
    """
    print("\n=== SEC EDGAR FILINGS ===")
    stats: Dict[str, int] = {}

    for i, ticker in enumerate(EDGAR_TICKERS, 1):
        try:
            cik = _sec_get_cik(ticker)
            if not cik:
                stats[ticker] = 0
                continue

            # Get filings index
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            raw = _safe_request(url, timeout=20, headers=SEC_HEADERS)
            if not raw:
                stats[ticker] = 0
                continue
            data = json.loads(raw.decode("utf-8"))

            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            ticker_dir = TRANSCRIPTS_DIR / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            saved = 0

            for j in range(min(len(forms), 20)):  # Last 20 filings
                form = forms[j]
                if form not in ("10-K", "10-Q", "8-K"):
                    continue

                filing_date = dates[j] if j < len(dates) else "unknown"
                accession = accessions[j].replace("-", "") if j < len(accessions) else ""
                doc = primary_docs[j] if j < len(primary_docs) else ""

                if not accession or not doc:
                    continue

                # Save filing metadata
                meta = {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": filing_date,
                    "accession": accessions[j] if j < len(accessions) else "",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{doc}",
                }

                safe_date = filing_date.replace("-", "")
                meta_path = ticker_dir / f"{form}_{safe_date}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                saved += 1

            stats[ticker] = saved
            time.sleep(0.12)  # SEC rate limit: 10 req/sec

        except Exception as e:
            stats[ticker] = 0

        if i % 10 == 0:
            print(f"  SEC: {i}/{len(EDGAR_TICKERS)} done ...")

    ok = sum(1 for v in stats.values() if v > 0)
    print(f"  SEC EDGAR: {ok}/{len(EDGAR_TICKERS)} tickers with filings")
    return stats


# ──────────────────── NEWS (RSS feeds) ────────────────────

def _parse_rss_simple(xml_bytes: bytes) -> List[Dict[str, str]]:
    """Minimal RSS/Atom parser without xml.etree (handles broken feeds)."""
    text = xml_bytes.decode("utf-8", errors="replace")
    items: List[Dict[str, str]] = []

    # Find all <item> or <entry> blocks
    pattern = re.compile(r"<(?:item|entry)\b[^>]*>(.*?)</(?:item|entry)>", re.DOTALL | re.IGNORECASE)
    for block in pattern.findall(text):
        item: Dict[str, str] = {}
        for tag in ["title", "link", "pubDate", "published", "updated", "description", "summary", "author"]:
            match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
            if match:
                val = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", match.group(1), flags=re.DOTALL)
                val = re.sub(r"<[^>]+>", "", val).strip()
                item[tag] = val[:2000]  # Truncate very long descriptions
            # Also check for <link href="..."/>
            if tag == "link" and "link" not in item:
                link_match = re.search(r'<link[^>]+href=["\']([^"\']+)["\']', block, re.IGNORECASE)
                if link_match:
                    item["link"] = link_match.group(1)
        if item.get("title"):
            # Normalize date field
            item["date"] = item.get("pubDate") or item.get("published") or item.get("updated") or ""
            items.append(item)

    return items


def collect_news() -> Dict[str, int]:
    """Download financial news from RSS feeds."""
    print("\n=== FINANCIAL NEWS (RSS) ===")
    stats: Dict[str, int] = {}

    for feed_name, url in NEWS_RSS_FEEDS.items():
        try:
            raw = _safe_request(url, timeout=15)
            if not raw:
                stats[feed_name] = 0
                continue

            articles = _parse_rss_simple(raw)

            # Save as JSON lines
            out_path = NEWS_DIR / "rss" / f"{feed_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for art in articles:
                    art["source"] = feed_name
                    f.write(json.dumps(art, ensure_ascii=False) + "\n")

            stats[feed_name] = len(articles)

        except Exception:
            stats[feed_name] = 0

    ok = sum(1 for v in stats.values() if v > 0)
    print(f"  News RSS: {ok}/{len(NEWS_RSS_FEEDS)} feeds downloaded, {sum(stats.values())} total articles")
    return stats


# ──────────────────── CENTRAL BANK ────────────────────

def collect_central_bank() -> Dict[str, int]:
    """
    Download Fed speeches, FOMC statements and minutes from the Fed RSS feed
    and FRASER (Federal Reserve Archive).
    """
    print("\n=== CENTRAL BANK DATA ===")
    stats: Dict[str, int] = {}

    # 1. Fed speeches RSS
    try:
        raw = _safe_request("https://www.federalreserve.gov/feeds/speeches.xml", timeout=15)
        if raw:
            speeches = _parse_rss_simple(raw)
            out_path = CENTRALBANK_DIR / "fed_speeches" / "speeches.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in speeches:
                    s["source"] = "fed_speeches"
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            stats["fed_speeches"] = len(speeches)
        else:
            stats["fed_speeches"] = 0
    except Exception:
        stats["fed_speeches"] = 0

    # 2. FOMC press releases (monetary policy)
    try:
        raw = _safe_request("https://www.federalreserve.gov/feeds/press_monetary.xml", timeout=15)
        if raw:
            statements = _parse_rss_simple(raw)
            out_path = CENTRALBANK_DIR / "fomc_statements" / "monetary_policy.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in statements:
                    s["source"] = "fomc_monetary"
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            stats["fomc_statements"] = len(statements)
        else:
            stats["fomc_statements"] = 0
    except Exception:
        stats["fomc_statements"] = 0

    # 3. FOMC minutes (Beige Book, minutes via RSS)
    for feed_name, url in [
        ("fomc_minutes", "https://www.federalreserve.gov/feeds/press_other.xml"),
        ("fed_testimony", "https://www.federalreserve.gov/feeds/testimony.xml"),
    ]:
        try:
            raw = _safe_request(url, timeout=15)
            if raw:
                items = _parse_rss_simple(raw)
                out_path = CENTRALBANK_DIR / "fomc_minutes" / f"{feed_name}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for s in items:
                        s["source"] = feed_name
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                stats[feed_name] = len(items)
            else:
                stats[feed_name] = 0
        except Exception:
            stats[feed_name] = 0

    # 4. FOMC calendar (rate decisions with dates)
    try:
        fomc_dates = _build_fomc_calendar()
        if fomc_dates:
            out_path = CENTRALBANK_DIR / "fomc_calendar.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(fomc_dates, f, indent=2)
            stats["fomc_calendar"] = len(fomc_dates)
        else:
            stats["fomc_calendar"] = 0
    except Exception:
        stats["fomc_calendar"] = 0

    total = sum(stats.values())
    print(f"  Central Bank: {total} total items across {sum(1 for v in stats.values() if v > 0)} categories")
    return stats


def _build_fomc_calendar() -> List[Dict[str, str]]:
    """Build FOMC meeting dates from known schedule (2000-2026)."""
    # Historical FOMC dates are well-known, hardcode recent ones
    meetings = []
    # 2024-2026 FOMC scheduled meetings
    dates_2024 = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    ]
    dates_2025 = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    ]
    dates_2026 = [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    ]
    for d in dates_2024 + dates_2025 + dates_2026:
        meetings.append({"date": d, "type": "FOMC", "source": "federal_reserve_schedule"})
    return meetings


# ══════════════════════════════════════════════════════════════
#  MANIFEST & MAIN
# ══════════════════════════════════════════════════════════════

def write_manifest(all_stats: Dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_root": str(DATA),
        "structure": {
            "macro": str(MACRO_DIR),
            "markets": str(MARKETS_DIR),
            "stocks": str(STOCKS_DIR),
            "fundamentals": str(FUND_DIR),
            "news": str(NEWS_DIR),
            "transcripts": str(TRANSCRIPTS_DIR),
            "central_bank": str(CENTRALBANK_DIR),
        },
        "stats": all_stats,
    }
    with open(DATA / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest saved to {DATA / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ViziGenesis Data Collector")
    parser.add_argument("--section", choices=["macro", "markets", "stocks", "fundamentals",
                                                "news", "central_bank", "sec", "all"],
                        default="all", help="Which section to download")
    args = parser.parse_args()

    ensure_dirs()
    all_stats: Dict[str, Any] = {}

    started = datetime.utcnow()
    print(f"ViziGenesis Data Collection started at {started.isoformat()}")
    print(f"Output: {DATA}")

    if args.section in ("macro", "all"):
        all_stats["macro"] = collect_macro()

    if args.section in ("markets", "all"):
        all_stats["markets"] = collect_markets()

    if args.section in ("stocks", "all"):
        all_stats["stocks"] = collect_stocks()
        all_stats["indices"] = collect_indices()

    if args.section in ("fundamentals", "all"):
        all_stats["fundamentals"] = collect_fundamentals()

    if args.section in ("sec", "all"):
        all_stats["sec_filings"] = collect_sec_filings()

    if args.section in ("news", "all"):
        all_stats["news"] = collect_news()

    if args.section in ("central_bank", "all"):
        all_stats["central_bank"] = collect_central_bank()

    elapsed = (datetime.utcnow() - started).total_seconds()
    all_stats["elapsed_seconds"] = round(elapsed, 1)

    write_manifest(all_stats)

    print(f"\n{'='*60}")
    print(f"Data collection complete in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
