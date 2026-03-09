#!/usr/bin/env python3
"""
ViziGenesis — Instruction-Tuning Data Generator
================================================
Generates high-quality Q&A pairs from collected financial data
for fine-tuning an LLM on financial analysis tasks.

Output format: JSONL with {"instruction", "input", "output"} fields
compatible with Alpaca/ShareGPT format for QLoRA fine-tuning.

Run:  python data/generate_instructions.py
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
OUTPUT = PROCESSED / "instruction_data"

random.seed(42)


# ══════════════════════════════════════════════════════════════
#  1. MACRO Q&A GENERATION
# ══════════════════════════════════════════════════════════════

def _load_fred_series(name: str) -> Optional[pd.DataFrame]:
    path = DATA / "macro" / "fred" / f"{name}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if len(df.columns) >= 2:
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_values("date").reset_index(drop=True)
            return df if len(df) >= 5 else None
    except Exception:
        return None


def generate_macro_qa() -> List[Dict[str, str]]:
    """Generate Q&A pairs from macro data."""
    pairs: List[Dict[str, str]] = []

    # Load key macro series
    series_map = {}
    fred_dir = DATA / "macro" / "fred"
    if fred_dir.exists():
        for csv_path in fred_dir.glob("*.csv"):
            name = csv_path.stem
            df = _load_fred_series(name)
            if df is not None:
                series_map[name] = df

    if not series_map:
        return pairs

    # Template: "What is the current value of X?"
    for name, df in series_map.items():
        latest = df.iloc[-1]
        readable_name = name.replace("_", " ").title()
        pairs.append({
            "instruction": f"What is the latest value of {readable_name}?",
            "input": "",
            "output": (
                f"The latest {readable_name} is {latest['value']:.4g} "
                f"as of {latest['date']:%Y-%m-%d}."
            ),
        })

    # Template: "How has X changed recently?"
    for name, df in series_map.items():
        if len(df) < 12:
            continue
        latest = df.iloc[-1]
        prev_month = df.iloc[-2] if len(df) > 1 else latest
        prev_year = df.iloc[-12] if len(df) > 12 else df.iloc[0]

        change_m = latest["value"] - prev_month["value"]
        change_y = latest["value"] - prev_year["value"]
        readable = name.replace("_", " ").title()

        pairs.append({
            "instruction": f"How has {readable} changed recently?",
            "input": "",
            "output": (
                f"{readable} is currently at {latest['value']:.4g}. "
                f"It changed by {change_m:+.4g} from the previous reading ({prev_month['value']:.4g}). "
                f"Compared to a year ago ({prev_year['value']:.4g}), it has changed by {change_y:+.4g}."
            ),
        })

    # Template: "What happens to markets when CPI rises?"
    macro_impact_qa = [
        {
            "instruction": "If CPI rises 0.5%, how will the S&P 500 react?",
            "input": "",
            "output": (
                "A 0.5% rise in CPI is significantly above the Fed's 2% annual target and would likely "
                "trigger a negative market reaction. The S&P 500 could drop 1-3% in the short term as "
                "investors price in: 1) Higher-for-longer interest rates from the Fed, 2) Reduced "
                "corporate profit margins due to input cost inflation, 3) Lower consumer spending power. "
                "Growth/tech stocks (NASDAQ) would be hit harder than value stocks due to higher duration "
                "risk. Defensive sectors like utilities and consumer staples may outperform. Bond yields "
                "would rise (prices fall), and the DXY would strengthen. Consider: reducing tech exposure, "
                "adding energy/commodity positions, and monitoring Fed Funds futures for rate expectations."
            ),
        },
        {
            "instruction": "What happens to the stock market when unemployment rises sharply?",
            "input": "",
            "output": (
                "A sharp rise in unemployment (triggering the Sahm Rule — 0.5% increase from 12-month low) "
                "historically signals recession onset. Market implications: 1) S&P 500 typically falls "
                "15-25% during recessions, 2) Defensive sectors (healthcare, utilities, staples) outperform, "
                "3) Cyclical sectors (industrials, materials, discretionary) underperform, 4) Fed likely "
                "responds with rate cuts → positive for bonds, 5) Credit spreads widen as default risk "
                "increases. Trading strategy: reduce equity exposure, increase Treasury allocation, "
                "consider put protection on cyclical holdings."
            ),
        },
        {
            "instruction": "The Fed raised rates by 25bps. What sectors benefit and which are hurt?",
            "input": "",
            "output": (
                "A 25bps Fed rate hike has differentiated sector impacts:\n\n"
                "BENEFICIARIES:\n"
                "- Financials (JPM, BAC, GS): Wider net interest margins boost bank profitability\n"
                "- Insurance: Higher yields on investment portfolios\n"
                "- Short-duration value stocks: Less sensitive to discount rate changes\n\n"
                "HEADWINDS:\n"
                "- Tech/Growth (NVDA, AAPL): Higher discount rates reduce present value of future cash flows\n"
                "- REITs (AMT, PLD, O): Higher rates increase borrowing costs and make dividends less attractive\n"
                "- Utilities (NEE, DUK): Bond-proxy sectors face competition from higher yielding treasuries\n"
                "- Homebuilders/Housing: Higher mortgage rates reduce demand\n\n"
                "The actual market reaction depends heavily on forward guidance and whether the hike was expected."
            ),
        },
        {
            "instruction": "How does the yield curve inversion predict recessions?",
            "input": "",
            "output": (
                "The yield curve (10Y-2Y Treasury spread) has inverted before every US recession since 1970 "
                "with only one false positive. When it inverts (goes negative), it means short-term rates "
                "exceed long-term rates, signaling that:\n\n"
                "1. Markets expect the Fed will need to CUT rates due to economic weakness\n"
                "2. Banks face squeezed lending margins (borrow short, lend long)\n"
                "3. Investors are fleeing to long-term safety (buying 10Y bonds → lower yields)\n\n"
                "Timing: Recession typically arrives 12-24 months AFTER the initial inversion. The "
                "un-inversion (steepening) is often the more actionable signal — it means the Fed "
                "has started cutting, and recession is imminent.\n\n"
                "Current status: Check the T10Y2Y spread. If negative → recession risk is elevated. "
                "Investment implication: Increase cash/bond allocation, reduce cyclical equity exposure."
            ),
        },
        {
            "instruction": "What is the relationship between the US Dollar (DXY) and emerging market stocks?",
            "input": "",
            "output": (
                "The US Dollar Index (DXY) is inversely correlated with emerging market (EM) performance:\n\n"
                "STRONG DOLLAR (rising DXY) → NEGATIVE for EM:\n"
                "- EM countries with USD-denominated debt face higher repayment costs\n"
                "- Capital flows from EM to US as higher US rates attract investment\n"
                "- Commodities (priced in USD) become more expensive for foreign buyers\n"
                "- EM central banks may need to raise rates defensively to protect currencies\n\n"
                "WEAK DOLLAR (falling DXY) → POSITIVE for EM:\n"
                "- Capital flows into EM seeking higher returns\n"
                "- Commodity exporters benefit from USD weakness\n"
                "- Debt servicing costs decrease\n\n"
                "Example: When DXY fell from 114 to 100 in late 2022, EEM (EM ETF) rallied 25%.\n"
                "Trading: Long EEM/VWO when DXY breaks down, short/hedge when DXY strengthens above resistance."
            ),
        },
        {
            "instruction": "What does the VIX tell us about market conditions?",
            "input": "",
            "output": (
                "The VIX (CBOE Volatility Index) measures expected 30-day S&P 500 volatility:\n\n"
                "LEVELS:\n"
                "- VIX < 12: Extreme complacency (often precedes corrections)\n"
                "- VIX 12-18: Low volatility, bull market conditions\n"
                "- VIX 18-25: Normal/slightly elevated\n"
                "- VIX 25-35: Fear, potential buying opportunity for contrarian investors\n"
                "- VIX > 35: Panic (2020 COVID peak: ~82, 2008 crisis peak: ~80)\n\n"
                "KEY PROPERTIES:\n"
                "- Mean-reverting: Extreme VIX readings typically don't persist\n"
                "- VIX > 40 often marks market bottoms (sell fear, buy opportunity)\n"
                "- VIX term structure: Contango (normal) vs Backwardation (stress signal)\n"
                "- VVIX (vol of vol): measures uncertainty about future volatility\n\n"
                "Trading: Sell vol (short VXX/UVXY) when VIX spikes above 30. Buy put protection when VIX < 13."
            ),
        },
        {
            "instruction": "How does quantitative tightening (QT) affect the stock market?",
            "input": "",
            "output": (
                "Quantitative Tightening (QT) is when the Fed reduces its balance sheet by letting bonds "
                "mature without reinvesting. This drains liquidity from the financial system:\n\n"
                "MECHANISM:\n"
                "1. Fed balance sheet (WALCL) shrinks → less money in the system\n"
                "2. Banks hold fewer excess reserves → tighter lending conditions\n"
                "3. Treasury supply increases → higher yields → headwind for stocks\n\n"
                "NET LIQUIDITY FORMULA:\n"
                "Net Liquidity = Fed Balance Sheet - Treasury General Account (TGA) - Reverse Repo (RRP)\n\n"
                "This metric is highly correlated with S&P 500 performance. When net liquidity rises, "
                "stocks tend to rise, and vice versa. QT at $95B/month (2022-2024) was a significant "
                "drag on risk assets.\n\n"
                "Trading: Monitor WALCL, TGA (WTREGEN), and RRP (RRPONTSYD) on FRED. Rising net liquidity "
                "= bullish for equities. Declining = bearish, especially for high-beta/growth names."
            ),
        },
    ]
    pairs.extend(macro_impact_qa)

    return pairs


# ══════════════════════════════════════════════════════════════
#  2. STOCK ANALYSIS Q&A GENERATION
# ══════════════════════════════════════════════════════════════

def generate_stock_qa() -> List[Dict[str, str]]:
    """Generate Q&A pairs from stock data and fundamentals."""
    pairs: List[Dict[str, str]] = []

    # Load stock summaries
    summaries_path = PROCESSED / "stock_summaries.jsonl"
    fund_path = PROCESSED / "fundamentals_summaries.jsonl"

    stocks: Dict[str, Dict] = {}
    if summaries_path.exists():
        with open(summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                    stocks[s["symbol"]] = s
                except Exception:
                    pass

    fundamentals: Dict[str, Dict] = {}
    if fund_path.exists():
        with open(fund_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                    fundamentals[s["symbol"]] = s
                except Exception:
                    pass

    if not stocks:
        return pairs

    # Template: "Analyze [STOCK]'s recent performance"
    for sym, info in stocks.items():
        rets = info.get("returns", {})
        ret_str = ", ".join(f"{k}: {v:+.1f}%" for k, v in rets.items()) if rets else "N/A"
        fund_info = fundamentals.get(sym, {})
        fund_text = fund_info.get("text", "")

        pairs.append({
            "instruction": f"Analyze {sym}'s recent stock performance.",
            "input": "",
            "output": (
                f"{sym} is trading at ${info.get('latest_price', 0):.2f}. "
                f"Daily return: {info.get('daily_return_pct', 0):+.2f}%. "
                f"Period returns: {ret_str}. "
                f"52-week range: ${info.get('52w_low', 0):.2f} - ${info.get('52w_high', 0):.2f}. "
                f"Average 20-day volume: {info.get('avg_volume_20d', 0):,}. "
                f"{fund_text if fund_text else ''}"
            ),
        })

    # Complex scenario Q&A
    scenario_qa = [
        {
            "instruction": "NVDA earnings are strong but the tech market is weak. What's the trading opportunity?",
            "input": "",
            "output": (
                "This is a classic divergence scenario. Strong NVDA earnings in a weak tech environment:\n\n"
                "BULLISH CASE:\n"
                "- NVDA's AI/datacenter revenue validates the AI capex cycle\n"
                "- Strong earnings = fundamental support regardless of sector weakness\n"
                "- Relative strength in a weak market is a positive signal\n"
                "- If guidance is strong, institutional buyers will accumulate\n\n"
                "BEARISH RISKS:\n"
                "- Even great companies fall in bear markets (correlation goes to 1 in crisis)\n"
                "- High valuation means any guidance miss will be punished severely\n"
                "- Tech sector selling pressure may overwhelm individual strength\n"
                "- Watch QQQ/XLK for sector-level support/resistance\n\n"
                "TRADING STRATEGY:\n"
                "1. If NVDA pulls back to 20-day SMA on sector weakness → accumulate position\n"
                "2. Use a pairs trade: Long NVDA / Short weak tech (e.g., INTC) to isolate alpha\n"
                "3. Buy call spreads to limit downside risk while capturing upside\n"
                "4. Monitor SOXX (semiconductor index) for sector confirmation\n"
                "5. Key levels: Track NVDA's 50-day and 200-day SMAs as support"
            ),
        },
        {
            "instruction": "Should I buy or sell when a stock gaps up 10% on earnings?",
            "input": "",
            "output": (
                "A 10% earnings gap requires careful analysis:\n\n"
                "HOLD/ADD (gap fills are rare) when:\n"
                "- Revenue AND EPS both beat by 5%+\n"
                "- Forward guidance raised significantly\n"
                "- Volume is 3x+ average (institutional buying)\n"
                "- The gap breaks above a key resistance level (all-time high, multi-month range)\n"
                "- RSI was below 70 before the gap\n\n"
                "SELL/TRIM when:\n"
                "- EPS beat but revenue missed (earnings quality concern)\n"
                "- Stock already up 50%+ in 3 months before the gap\n"
                "- Volume is lower than expected (lack of conviction)\n"
                "- Gap into heavy resistance (prior all-time high)\n"
                "- Broader market is in a downtrend (QQQ/SPY below 50-day SMA)\n\n"
                "GENERAL RULE:\n"
                "Never chase the open. Wait 30 minutes for price discovery. If the stock holds "
                "gains and closes strong, it's likely to continue. Gaps that fill in the first "
                "hour are warning signs."
            ),
        },
        {
            "instruction": "Compare the investment thesis for AAPL vs MSFT.",
            "input": "",
            "output": (
                "AAPL (Apple) vs MSFT (Microsoft) — Two different mega-cap tech strategies:\n\n"
                "AAPL THESIS:\n"
                "- Hardware ecosystem + services revenue (App Store, iCloud, Apple Music)\n"
                "- Massive installed base: 2B+ active devices creates recurring revenue\n"
                "- Services segment growing faster (higher margins, 70%+ gross)\n"
                "- Buyback machine: $80-90B/year in share repurchases\n"
                "- Risks: iPhone cycle dependency, China revenue exposure (~20%), AI integration execution\n\n"
                "MSFT THESIS:\n"
                "- Cloud leader: Azure growing 25%+ YoY, gaining share vs AWS\n"
                "- AI first-mover: OpenAI partnership, Copilot integration across Office 365\n"
                "- Enterprise sticky: 365 subscription model with high retention\n"
                "- Gaming: Activision acquisition adds recurring revenue\n"
                "- Risks: Antitrust scrutiny, cloud growth deceleration, AI monetization uncertainty\n\n"
                "VERDICT:\n"
                "MSFT has stronger growth vectors (AI, cloud) and higher revenue diversification. "
                "AAPL is the better capital return story (buybacks, dividends). "
                "For growth: MSFT. For safety/yield: AAPL. Both are core holdings."
            ),
        },
        {
            "instruction": "What stocks benefit from falling interest rates?",
            "input": "",
            "output": (
                "Falling interest rates create clear sector winners:\n\n"
                "STRONGEST BENEFICIARIES:\n"
                "1. Growth/Tech stocks (NVDA, AAPL, MSFT, GOOGL): Lower discount rates increase "
                "present value of future cash flows. High-duration assets benefit most.\n"
                "2. REITs (AMT, PLD, EQIX, O): Lower borrowing costs improve profitability, "
                "higher dividend yields become more attractive vs. lower bond yields.\n"
                "3. Utilities (NEE, DUK, SO): Bond-proxy sectors benefit from relative yield advantage.\n"
                "4. Homebuilders (DHI, LEN, TOL): Lower mortgage rates stimulate housing demand.\n"
                "5. Small-caps (IWM): More leveraged companies benefit from lower debt servicing costs.\n\n"
                "MODERATE BENEFICIARIES:\n"
                "- Consumer Discretionary: Lower rates boost consumer spending/borrowing\n"
                "- Biotech: Cash-burning companies get cheaper financing\n\n"
                "LOSERS:\n"
                "- Banks (JPM, BAC): Net interest margins compress\n"
                "- Insurance: Investment income declines\n"
                "- USD-denominated assets: Dollar weakens as interest rate differential narrows"
            ),
        },
    ]
    pairs.extend(scenario_qa)

    return pairs


# ══════════════════════════════════════════════════════════════
#  3. CROSS-ASSET / MULTI-FACTOR Q&A
# ══════════════════════════════════════════════════════════════

def generate_cross_asset_qa() -> List[Dict[str, str]]:
    """Generate complex multi-factor analysis Q&A pairs."""
    pairs = [
        {
            "instruction": "Oil prices are spiking while the economy is slowing. What's the investment strategy?",
            "input": "",
            "output": (
                "This is a stagflationary scenario — one of the worst for traditional portfolios:\n\n"
                "SITUATION: Rising oil + slowing growth = cost-push inflation + margin compression\n\n"
                "STRATEGY:\n"
                "1. ENERGY OVERWEIGHT: XOM, CVX, COP benefit directly from higher prices\n"
                "2. COMMODITY DIVERSIFICATION: Gold (GLD) as inflation hedge, agricultural commodities\n"
                "3. REDUCE: Consumer discretionary (compressed margins), airlines, transports\n"
                "4. SHORT-DURATION FIXED INCOME: SHY, BIL — avoid long-duration in rising rate environment\n"
                "5. DEFENSIVE EQUITY: Healthcare (UNH, JNJ), Staples (PG, KO, PEP)\n"
                "6. INFLATION PROTECTION: TIP (TIPS ETF), I-bonds\n"
                "7. CASH: Hold elevated cash position (5-15%) for opportunities\n\n"
                "AVOID: High-growth tech (high duration), REITs (rate sensitive), small-caps (leverage risk)\n\n"
                "HISTORICAL PARALLEL: 1970s stagflation — energy and gold outperformed, equities flat for a decade."
            ),
        },
        {
            "instruction": "China's PMI data came in weak. How does this affect global markets?",
            "input": "",
            "output": (
                "Weak China PMI has cascading effects across global markets:\n\n"
                "DIRECT IMPACTS:\n"
                "- Commodities: Copper (-2-4%), iron ore (-3-5%), oil (-1-2%) as China is the largest consumer\n"
                "- Australia (EWA, AUD): Mining sector hit, AUD weakens vs USD\n"
                "- Emerging Markets (EEM, VWO): Sell-off on growth concerns\n"
                "- European luxury (LVMH, Hermes): Chinese consumer spending is a key driver\n\n"
                "SECONDARY EFFECTS:\n"
                "- Global growth expectations revised down\n"
                "- Commodity exporters (Brazil, Chile, Australia) currencies weaken\n"
                "- Safe haven flows: USD, JPY, Treasuries rally\n"
                "- Supply chain stocks may benefit if China stimulates (cheap production)\n\n"
                "POTENTIAL RESPONSE:\n"
                "- If PBOC cuts RRR or rates → bullish reversal for EM and commodities\n"
                "- If no stimulus → continued weakness, de-risking continues\n\n"
                "TRADES: Short copper (HG=F), long USD/AUD (AUDUSD=X), monitor FXI for base formation. "
                "If China announces stimulus, reverse positioning aggressively."
            ),
        },
        {
            "instruction": "The Fed just announced a pause in rate hikes. What should I do with my portfolio?",
            "input": "",
            "output": (
                "A Fed pause is historically a positive inflection point for markets:\n\n"
                "WHAT A PAUSE SIGNALS:\n"
                "- The Fed believes current rates are restrictive enough to slow inflation\n"
                "- Next move is likely a CUT (historically within 6-12 months after last hike)\n"
                "- Peak rates are in → duration risk decreases for bonds\n\n"
                "PORTFOLIO ADJUSTMENTS:\n"
                "1. INCREASE GROWTH EXPOSURE: Add NASDAQ/QQQ, SOXX — growth stocks outperform "
                "in the 6 months following a Fed pause (avg +15% historical return)\n"
                "2. ADD DURATION: Buy TLT/IEF — if next move is a cut, long bonds will rally\n"
                "3. SMALL-CAP TILT: IWM tends to outperform SPY following pauses\n"
                "4. REDUCE CASH: Deploy cash into equities — sitting in money market becomes less "
                "attractive as short-term rates are expected to decline\n"
                "5. CREDIT: HYG/JNK may rally as default risk decreases with easier policy ahead\n\n"
                "RISKS TO MONITOR:\n"
                "- If inflation re-accelerates, the pause becomes temporary (hawkish repricing)\n"
                "- If unemployment rises fast, the pause → cuts happen faster than expected (recession)\n"
                "- Watch Fed Funds futures (CME FedWatch) for rate cut probability."
            ),
        },
        {
            "instruction": "How do I build a recession-proof portfolio?",
            "input": "",
            "output": (
                "A recession-proof portfolio prioritizes capital preservation and defensive positioning:\n\n"
                "ALLOCATION (example):\n"
                "- 25% Treasuries (TLT, IEF): Flight-to-safety, rates fall in recession\n"
                "- 20% Healthcare (XLV, UNH, JNJ): Non-cyclical demand, stable earnings\n"
                "- 15% Consumer Staples (XLP, PG, KO, WMT): People still buy essentials\n"
                "- 10% Utilities (XLU, NEE, DUK): Regulated revenue, high dividends\n"
                "- 10% Gold (GLD): Safe haven, central banks buy during uncertainty\n"
                "- 10% Cash/Short-term Treasuries (SHY, BIL): Optionality to buy the dip\n"
                "- 5% Dividend Aristocrats (NOBL): Companies that raised dividends 25+ years straight\n"
                "- 5% Inverse/Hedge (SH, or put options on SPY): Direct downside protection\n\n"
                "KEY PRINCIPLES:\n"
                "- Avoid: Highly leveraged companies, cyclicals (industrials, materials), small-caps\n"
                "- Quality matters: Strong balance sheets, positive free cash flow, low debt/equity\n"
                "- Dividend sustainability: Payout ratio below 60%, growing FCF\n"
                "- Rebalance monthly: If stocks fall 20%+, start rotating from bonds → equities gradually"
            ),
        },
        {
            "instruction": "What does a strengthening Japanese Yen (falling USDJPY) mean for global markets?",
            "input": "",
            "output": (
                "A strengthening Yen (falling USDJPY) has significant global implications:\n\n"
                "CAUSES:\n"
                "- Bank of Japan (BOJ) hawkish shift (raising rates, ending yield curve control)\n"
                "- US-Japan interest rate differential narrowing\n"
                "- Global risk-off (JPY is a safe-haven carry trade currency)\n\n"
                "MARKET IMPACTS:\n"
                "1. JPY CARRY TRADE UNWIND: Global selloff risk. Trillions in JPY-funded leveraged "
                "positions (borrowing cheap JPY to buy high-yield assets) must be unwound. This caused "
                "the August 2024 crash (Nikkei -12% in one day, VIX spike to 65).\n"
                "2. JAPANESE EQUITIES (EWJ, Nikkei): Mixed — exporters (Toyota, Sony) hurt by strong "
                "JPY reducing overseas earnings; domestic-focused stocks may benefit.\n"
                "3. GLOBAL BONDS: If cause is BOJ selling US Treasuries → UST yields spike\n"
                "4. RISK ASSETS: General de-risking as leveraged positions unwind\n\n"
                "TRADING:\n"
                "- If USDJPY drops sharply (>3% in a week): Go defensive, reduce leverage\n"
                "- Monitor VIX for carry trade unwind stress\n"
                "- Long JPY/short EM currencies as a risk hedge"
            ),
        },
        {
            "instruction": "Analyze the Vietnam stock market (VN-Index) and its key drivers.",
            "input": "",
            "output": (
                "The VN-Index is Vietnam's main benchmark, covering ~400 listed companies:\n\n"
                "TOP SECTORS BY WEIGHT:\n"
                "1. Banking (~35%): VCB, TCB, CTG, BID, MBB, ACB — dominate the index\n"
                "2. Real Estate (~15%): VIC, VHM, NVL — cyclical and rate-sensitive\n"
                "3. Steel/Materials (~5%): HPG — largest steelmaker, proxy for construction activity\n"
                "4. Retail/Consumer: MWG (Mobile World), PNJ (jewelry), MSN (Masan Group)\n"
                "5. Tech: FPT — Vietnam's largest tech company, growing IT outsourcing\n\n"
                "KEY DRIVERS:\n"
                "- State Bank of Vietnam (SBV) policy rates: Rate cuts → positive for banks and real estate\n"
                "- USD/VND: Strong VND attracts foreign investment; VND depreciation triggers outflows\n"
                "- Foreign institutional flows: FTSE/MSCI index inclusion catalysts\n"
                "- China+1: Manufacturing shift benefiting Vietnam (Samsung, Intel factories)\n"
                "- US interest rates: Higher US rates → capital outflows from VN\n"
                "- Real estate market health: Land prices, development regulations\n\n"
                "RISKS: Corporate governance concerns, liquidity limitations, margin lending concentration\n"
                "OPPORTUNITY: GDP growth 6-7%, young demographics, FDI inflows, potential MSCI EM inclusion"
            ),
        },
    ]
    return pairs


# ══════════════════════════════════════════════════════════════
#  4. FINANCIAL LITERACY Q&A
# ══════════════════════════════════════════════════════════════

def generate_financial_literacy_qa() -> List[Dict[str, str]]:
    """Generate educational Q&A about financial concepts."""
    return [
        {
            "instruction": "What is P/E ratio and how should I use it?",
            "input": "",
            "output": (
                "P/E (Price-to-Earnings) ratio = Stock Price / Earnings Per Share (EPS)\n\n"
                "TYPES:\n"
                "- Trailing P/E: Based on last 12 months actual earnings\n"
                "- Forward P/E: Based on analyst estimates for next 12 months\n\n"
                "INTERPRETATION:\n"
                "- S&P 500 average: ~20x historically, ~18x is 'fair value'\n"
                "- Growth stocks: 30-50x+ (NVDA, AMZN) — justified if earnings are growing fast\n"
                "- Value stocks: 8-15x (banks, energy) — slower growth, higher dividends\n"
                "- < 10x: Either cheap or in trouble (value trap risk)\n\n"
                "PEG RATIO (P/E ÷ Earnings Growth Rate) is more useful:\n"
                "- PEG < 1: Potentially undervalued relative to growth\n"
                "- PEG 1-2: Fairly valued\n"
                "- PEG > 2: Expensive relative to growth\n\n"
                "CAVEATS: P/E is meaningless for unprofitable companies, cyclical sectors (use P/B), "
                "or during earnings manipulation. Always compare within the same sector."
            ),
        },
        {
            "instruction": "Explain RSI (Relative Strength Index) and how to use it in trading.",
            "input": "",
            "output": (
                "RSI is a momentum oscillator measuring speed and magnitude of price changes (0-100):\n\n"
                "FORMULA: RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss over N periods\n"
                "Standard period: 14 days\n\n"
                "INTERPRETATION:\n"
                "- RSI > 70: Overbought (potential selling opportunity)\n"
                "- RSI < 30: Oversold (potential buying opportunity)\n"
                "- RSI 40-60: Neutral zone\n\n"
                "TRADING SIGNALS:\n"
                "1. Buy when RSI crosses ABOVE 30 from oversold territory\n"
                "2. Sell when RSI crosses BELOW 70 from overbought territory\n"
                "3. Bullish divergence: Price makes lower low but RSI makes higher low → reversal signal\n"
                "4. Bearish divergence: Price makes higher high but RSI makes lower high → weakness\n\n"
                "IMPORTANT CAVEATS:\n"
                "- In strong uptrends, RSI can stay above 70 for weeks (don't sell just because it's 'overbought')\n"
                "- In strong downtrends, RSI can stay below 30 for extended periods\n"
                "- Use RSI in combination with trend analysis (SMA 200) and volume confirmation\n"
                "- RSI(7) is more sensitive (good for short-term), RSI(14) is standard"
            ),
        },
        {
            "instruction": "What is MACD and how do I read it?",
            "input": "",
            "output": (
                "MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator:\n\n"
                "COMPONENTS:\n"
                "1. MACD Line = EMA(12) - EMA(26) of closing prices\n"
                "2. Signal Line = EMA(9) of the MACD Line\n"
                "3. Histogram = MACD Line - Signal Line\n\n"
                "SIGNALS:\n"
                "- BULLISH: MACD crosses ABOVE signal line → buy signal\n"
                "- BEARISH: MACD crosses BELOW signal line → sell signal\n"
                "- BULLISH DIVERGENCE: Price makes new low but MACD makes higher low → reversal incoming\n"
                "- Histogram growing: Momentum strengthening. Histogram shrinking: Momentum fading.\n\n"
                "BEST PRACTICES:\n"
                "- Combine with RSI for confirmation (MACD + RSI both bullish = strong signal)\n"
                "- Zero-line crosses are significant: MACD crossing above 0 = bullish trend confirmed\n"
                "- Works best in trending markets, less reliable in ranging/choppy markets\n"
                "- Weekly MACD is excellent for identifying longer-term trend changes"
            ),
        },
        {
            "instruction": "Explain the concept of market capitalization and why it matters.",
            "input": "",
            "output": (
                "Market Capitalization = Share Price × Total Shares Outstanding\n\n"
                "CATEGORIES:\n"
                "- Mega-cap: >$200B (AAPL at ~$3T, MSFT at ~$3T) — most stable, index movers\n"
                "- Large-cap: $10B-$200B — solid growth, institutional favorites\n"
                "- Mid-cap: $2B-$10B — growth potential with moderate risk\n"
                "- Small-cap: $300M-$2B — higher growth potential, higher volatility\n"
                "- Micro-cap: <$300M — speculative, liquidity risk\n\n"
                "WHY IT MATTERS:\n"
                "1. INDEX WEIGHTING: S&P 500 is market-cap weighted → AAPL/MSFT/NVDA dominate\n"
                "2. LIQUIDITY: Larger cap = more liquid = tighter bid-ask spreads\n"
                "3. RISK PROFILE: Larger cap = generally lower volatility and drawdown risk\n"
                "4. INSTITUTIONAL FLOWS: Many funds can only invest in stocks above certain market caps\n"
                "5. SECTOR ANALYSIS: Enterprise Value (EV) = Market Cap + Debt - Cash is more accurate\n\n"
                "IMPLICATIONS:\n"
                "- In risk-off: mega-caps outperform (flight to quality)\n"
                "- In risk-on/recovery: small-caps outperform (higher beta, more leverage)\n"
                "- Watch the Mag 7 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA) — their combined "
                "market cap is ~30% of S&P 500"
            ),
        },
        {
            "instruction": "How do Bollinger Bands work and how to trade with them?",
            "input": "",
            "output": (
                "Bollinger Bands measure price volatility using standard deviations around a moving average:\n\n"
                "COMPONENTS:\n"
                "- Middle Band: 20-period SMA\n"
                "- Upper Band: SMA + 2σ (standard deviations)\n"
                "- Lower Band: SMA - 2σ\n"
                "- Bandwidth: (Upper - Lower) / Middle\n\n"
                "KEY CONCEPTS:\n"
                "1. SQUEEZE: Bands narrow → low volatility → big move incoming (direction unknown)\n"
                "2. EXPANSION: Bands widen → trend in progress\n"
                "3. %B: (Price - Lower) / (Upper - Lower) — position within the bands\n\n"
                "TRADING STRATEGIES:\n"
                "- Mean Reversion: When price touches lower band with RSI < 30 → buy signal. "
                "When price touches upper band with RSI > 70 → sell signal.\n"
                "- Breakout: If price closes outside the bands on high volume → trend continuation\n"
                "- Walking the bands: In strong trends, price 'walks' along the upper/lower band\n\n"
                "IMPORTANT: Don't automatically sell when price hits upper band in an uptrend. "
                "Use confirmation (RSI divergence, volume decline, candlestick patterns)."
            ),
        },
    ]


# ══════════════════════════════════════════════════════════════
#  5. NEWS IMPACT Q&A (from collected news)
# ══════════════════════════════════════════════════════════════

def generate_news_qa() -> List[Dict[str, str]]:
    """Generate Q&A pairs from collected news data."""
    pairs: List[Dict[str, str]] = []

    news_path = PROCESSED / "news_corpus.jsonl"
    if not news_path.exists():
        return pairs

    articles: List[Dict] = []
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                articles.append(json.loads(line))
            except Exception:
                pass

    # Generate Q&A from news articles
    for art in articles[:200]:  # Limit to prevent explosion
        title = art.get("title", "")
        content = art.get("content", "")
        date = art.get("date", "")

        if not title or len(title) < 20:
            continue

        pairs.append({
            "instruction": f"What is the market implication of this news: \"{title}\"?",
            "input": content[:500] if content else "",
            "output": (
                f"This news headline from {date}: \"{title}\"\n\n"
                f"Potential market implications:\n"
                f"1. Identify affected sectors and stocks based on the topic\n"
                f"2. Consider whether this is positive or negative for earnings expectations\n"
                f"3. Evaluate if this represents a trend change or a transient event\n"
                f"4. Check if the market has already priced in this information\n"
                f"5. Look for second-order effects on related industries and trading partners\n\n"
                f"Key: News impact = Surprise Factor × Magnitude × Persistence"
            ),
        })

    return pairs


# ══════════════════════════════════════════════════════════════
#  6. COMPILE ALL INSTRUCTION DATA
# ══════════════════════════════════════════════════════════════

def compile_all() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Dict[str, str]] = []
    stats: Dict[str, int] = {}

    generators = [
        ("macro_qa", generate_macro_qa),
        ("stock_qa", generate_stock_qa),
        ("cross_asset_qa", generate_cross_asset_qa),
        ("financial_literacy", generate_financial_literacy_qa),
        ("news_qa", generate_news_qa),
    ]

    for name, gen_fn in generators:
        print(f"  Generating {name}...")
        pairs = gen_fn()
        stats[name] = len(pairs)
        all_pairs.extend(pairs)
        print(f"    → {len(pairs)} Q&A pairs")

    # Shuffle for training
    random.shuffle(all_pairs)

    # Split into train/eval
    n = len(all_pairs)
    split = int(n * 0.9)
    train = all_pairs[:split]
    val = all_pairs[split:]

    # Save in Alpaca format (compatible with most fine-tuning frameworks)
    for name, data in [("train", train), ("val", val)]:
        # JSONL format
        out_path = OUTPUT / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Also save as JSON array (some frameworks prefer this)
        out_path_json = OUTPUT / f"{name}.json"
        with open(out_path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Save in ShareGPT / conversation format (for chat fine-tuning)
    sharegpt_train = []
    for item in train:
        sharegpt_train.append({
            "conversations": [
                {"from": "human", "value": item["instruction"] + ("\n" + item["input"] if item.get("input") else "")},
                {"from": "gpt", "value": item["output"]},
            ]
        })
    with open(OUTPUT / "train_sharegpt.json", "w", encoding="utf-8") as f:
        json.dump(sharegpt_train, f, indent=2, ensure_ascii=False)

    stats["total"] = n
    stats["train"] = len(train)
    stats["val"] = len(val)

    with open(OUTPUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Instruction data generated:")
    print(f"  Total: {n} Q&A pairs")
    print(f"  Train: {len(train)}, Val: {len(val)}")
    print(f"  Output: {OUTPUT}")
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    compile_all()
