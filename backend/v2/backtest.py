"""
ViziGenesis V2 — Backtest Engine
===================================
Full-featured backtesting with:
  • Long / short / flat strategies
  • Kelly Criterion position sizing (capped)
  • Transaction costs + slippage
  • Trailing stop-loss
  • Max exposure & diversification rules
  • Crisis-period stress tests
  • Per-stock contribution analysis
  • Monthly PnL breakdown

Produces comprehensive backtest report ready for PDF rendering.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.v2.config import (
    BT_SIGNAL_LONG, BT_SIGNAL_SHORT, BT_KELLY_CAP,
    BT_MAX_EXPOSURE, BT_INITIAL_CAPITAL, BT_COMMISSION_BPS,
    BT_SLIPPAGE_BPS, BT_STOP_LOSS_PCT, CRISIS_PERIODS,
)

logger = logging.getLogger("vizigenesis.v2.backtest")


@dataclass
class Trade:
    date: str
    direction: str    # "LONG" or "SHORT"
    signal: float     # P(up) or P(down)
    size: float       # fraction of capital
    daily_return: float
    pnl: float        # realized P&L after costs
    cost: float       # transaction cost + slippage


@dataclass
class BacktestResult:
    equity_curve: List[float]
    dates: List[str]
    trades: List[Trade]
    metrics: Dict
    signals: List[Dict]
    monthly_pnl: Dict
    per_stock: Dict = field(default_factory=dict)
    stress_tests: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# 1.  Kelly Criterion sizing
# ═══════════════════════════════════════════════════════════════════════
def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cap: float = BT_KELLY_CAP,
    min_fraction: float = 0.02,
) -> float:
    """
    Kelly fraction = (p * b - q) / b
    where p = win_rate, b = avg_win / avg_loss, q = 1 - p
    """
    if avg_loss == 0 or avg_win == 0:
        return min_fraction
    b = abs(avg_win / avg_loss)
    q = 1.0 - win_rate
    f = (win_rate * b - q) / b
    return max(min(f, cap), min_fraction)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Main backtest engine
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(
    dates: pd.DatetimeIndex,
    daily_returns: np.ndarray,   # actual daily returns
    direction_probs: np.ndarray,  # calibrated P(up)
    expected_returns: Optional[np.ndarray] = None,  # expected 1d return (%)
    initial_capital: float = BT_INITIAL_CAPITAL,
    signal_long: float = BT_SIGNAL_LONG,
    signal_short: float = BT_SIGNAL_SHORT,
    kelly_cap: float = BT_KELLY_CAP,
    commission_bps: float = BT_COMMISSION_BPS,
    slippage_bps: float = BT_SLIPPAGE_BPS,
    stop_loss_pct: float = BT_STOP_LOSS_PCT,
    max_exposure: float = BT_MAX_EXPOSURE,
) -> BacktestResult:
    """
    Signal-based backtest with long/short/flat positions.

    Rules:
      • Long when P(up) > signal_long AND expected_return > 0 (if available)
      • Short when P(down) > signal_short
      • Flat otherwise
      • Position size: Kelly fraction, capped at kelly_cap
      • Exit on stop-loss or signal reversal
    """
    n = min(len(dates), len(daily_returns), len(direction_probs))
    dates = dates[:n]
    rets = daily_returns[:n]
    probs = direction_probs[:n]
    exp_rets = expected_returns[:n] if expected_returns is not None else None

    capital = initial_capital
    equity = [capital]
    trades = []
    signals = []

    # Rolling win/loss tracking for Kelly
    wins = []
    losses = []
    total_cost_bps = commission_bps + slippage_bps

    # Track position state
    position = "FLAT"
    entry_price_equiv = 0.0
    running_pnl = 0.0
    max_since_entry = 0.0

    for i in range(n):
        p_up = float(probs[i])
        p_down = 1.0 - p_up
        actual_ret = float(rets[i])

        # Compute Kelly fraction based on history
        if len(wins) >= 5 and len(losses) >= 2:
            wr = len(wins) / (len(wins) + len(losses))
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 1
            kf = kelly_fraction(wr, avg_w, avg_l, cap=kelly_cap)
        else:
            kf = kelly_cap * 0.5  # conservative until enough history

        # ── Signal logic ──────────────────────────────────────────────
        signal_dir = "FLAT"
        if p_up > signal_long:
            if exp_rets is not None:
                if exp_rets[i] > 0:
                    signal_dir = "LONG"
            else:
                signal_dir = "LONG"
        elif p_down > signal_short:
            signal_dir = "SHORT"

        # ── Execute ───────────────────────────────────────────────────
        if signal_dir != "FLAT":
            size = min(kf, max_exposure)
            cost = size * total_cost_bps / 10000

            if signal_dir == "LONG":
                trade_pnl = size * actual_ret - cost
            else:  # SHORT
                trade_pnl = size * (-actual_ret) - cost

            capital += capital * trade_pnl

            trades.append(Trade(
                date=str(dates[i].date()),
                direction=signal_dir,
                signal=p_up,
                size=size,
                daily_return=actual_ret,
                pnl=trade_pnl,
                cost=cost,
            ))

            if trade_pnl > 0:
                wins.append(trade_pnl)
            else:
                losses.append(abs(trade_pnl))

            # Track for trailing stop
            running_pnl += trade_pnl
            max_since_entry = max(max_since_entry, running_pnl)

            # Trailing stop check
            if max_since_entry - running_pnl > stop_loss_pct:
                signal_dir = "FLAT"
                running_pnl = 0
                max_since_entry = 0
        else:
            running_pnl = 0
            max_since_entry = 0

        equity.append(capital)
        signals.append({
            "date": str(dates[i].date()),
            "p_up": round(p_up, 4),
            "signal": signal_dir,
            "size": round(kf, 4),
            "actual_return": round(actual_ret, 6),
        })

    # ── Buy-and-hold benchmark ────────────────────────────────────────
    bah_equity = [initial_capital]
    bah_cap = initial_capital
    for r in rets:
        bah_cap *= (1 + r)
        bah_equity.append(bah_cap)

    metrics = compute_backtest_metrics(
        np.array(equity), np.array(bah_equity), trades, initial_capital
    )

    # Monthly PnL
    monthly = compute_monthly_pnl(dates, equity[1:])

    return BacktestResult(
        equity_curve=equity,
        dates=[str(d.date()) for d in dates],
        trades=trades,
        metrics=metrics,
        signals=signals,
        monthly_pnl=monthly,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3.  Metrics computation
# ═══════════════════════════════════════════════════════════════════════
def compute_backtest_metrics(
    equity: np.ndarray,
    benchmark_equity: np.ndarray,
    trades: List[Trade],
    initial_capital: float,
) -> Dict:
    """Comprehensive backtest metrics."""
    # Returns series
    equity_rets = np.diff(equity) / equity[:-1]
    equity_rets = equity_rets[np.isfinite(equity_rets)]
    bm_rets = np.diff(benchmark_equity) / benchmark_equity[:-1]
    bm_rets = bm_rets[np.isfinite(bm_rets)]

    n_days = len(equity_rets)
    if n_days == 0:
        return {"error": "no_data"}

    # Total return
    total_return = (equity[-1] / initial_capital - 1)
    bm_total = (benchmark_equity[-1] / initial_capital - 1) if len(benchmark_equity) > 0 else 0

    # Annualized return
    years = n_days / 252
    ann_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # Volatility
    ann_vol = np.std(equity_rets) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(np.min(drawdown))

    # Sharpe & Sortino
    daily_excess = equity_rets - 0.05 / 252  # risk-free ~5%
    sharpe = float(np.mean(daily_excess) / max(np.std(daily_excess), 1e-8) * np.sqrt(252))
    downside = equity_rets[equity_rets < 0]
    sortino_denom = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = float(np.mean(daily_excess) * np.sqrt(252) / max(sortino_denom, 1e-8))

    # Calmar
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0

    # Trade stats
    n_trades = len(trades)
    n_long = sum(1 for t in trades if t.direction == "LONG")
    n_short = sum(1 for t in trades if t.direction == "SHORT")
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0

    gross_profit = sum(t.pnl for t in winning)
    gross_loss = sum(abs(t.pnl) for t in losing)
    profit_factor = gross_profit / max(gross_loss, 1e-8) if gross_loss > 0 else float("inf")

    total_costs = sum(t.cost for t in trades)
    turnover = n_trades / max(n_days, 1)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualised_return_pct": round(ann_return * 100, 2),
        "annualised_volatility_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "total_costs_pct": round(total_costs * 100, 4),
        "turnover_per_day": round(turnover, 4),
        "benchmark_return_pct": round(bm_total * 100, 2),
        "alpha_pct": round((total_return - bm_total) * 100, 2),
        "final_equity": round(float(equity[-1]), 2),
        "n_trading_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4.  Monthly PnL breakdown
# ═══════════════════════════════════════════════════════════════════════
def compute_monthly_pnl(
    dates: pd.DatetimeIndex,
    equity: List[float],
) -> Dict:
    """Compute monthly returns."""
    df = pd.DataFrame({
        "date": dates[:len(equity)],
        "equity": equity,
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    monthly = df["equity"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    result = {}
    for dt, ret in monthly_ret.items():
        key = dt.strftime("%Y-%m")
        result[key] = round(float(ret) * 100, 2)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 5.  Stress tests on crisis periods
# ═══════════════════════════════════════════════════════════════════════
def run_stress_tests(
    full_result: BacktestResult,
    dates: pd.DatetimeIndex,
) -> Dict:
    """Evaluate backtest performance during crisis periods."""
    equity = np.array(full_result.equity_curve)
    date_strs = [str(d.date()) for d in dates]

    stress = {}
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = [(start <= d <= end) for d in date_strs]
        idx = [i for i, m in enumerate(mask) if m]

        if len(idx) < 10:
            stress[name] = {"status": "insufficient_data", "days": len(idx)}
            continue

        first, last = idx[0], idx[-1]
        period_equity = equity[first:last + 2]  # +2 because equity has one more element

        if len(period_equity) < 2:
            stress[name] = {"status": "too_short"}
            continue

        period_ret = (period_equity[-1] / period_equity[0] - 1) * 100
        peak = np.maximum.accumulate(period_equity)
        dd = (period_equity - peak) / peak
        max_dd = float(np.min(dd)) * 100

        n_trades = sum(
            1 for t in full_result.trades
            if start <= t.date <= end
        )

        stress[name] = {
            "period": f"{start} to {end}",
            "days": len(idx),
            "return_pct": round(period_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "n_trades": n_trades,
        }

    return stress


# ═══════════════════════════════════════════════════════════════════════
# 6.  Per-stock contribution (for panel backtest)
# ═══════════════════════════════════════════════════════════════════════
def compute_per_stock_contribution(
    trades: List[Trade],
    stock_symbols: Optional[List[str]] = None,
) -> Dict:
    """Compute PnL contribution per stock (if trade has stock info)."""
    # Group trades by stock (using date as proxy or a stock field)
    # In panel mode, each trade would carry the stock symbol
    # For now, return aggregate
    total_pnl = sum(t.pnl for t in trades)
    n_trades = len(trades)
    winning = sum(1 for t in trades if t.pnl > 0)

    return {
        "aggregate": {
            "total_pnl": round(total_pnl, 6),
            "n_trades": n_trades,
            "win_rate": round(winning / max(n_trades, 1) * 100, 2),
        }
    }
