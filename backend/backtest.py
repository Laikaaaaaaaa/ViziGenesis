"""
Backtest Engine for quant-mode signals.

Strategy:
  - Long when calibrated P(up) > threshold (default 0.60)
  - Cash (or flat) when P(up) <= threshold
  - Position sizing via Kelly Criterion, capped at 10 % of capital

Metrics:
  - Total Return, Annualised Return
  - Max Drawdown & Drawdown Duration
  - Sharpe Ratio (annualised, rf=0)
  - Sortino Ratio
  - Win Rate, Profit Factor
  - Calmar Ratio
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.config import (
    BT_SIGNAL_THRESHOLD, BT_KELLY_CAP,
    BT_INITIAL_CAPITAL, BT_COMMISSION_BPS,
)

logger = logging.getLogger("vizigenesis.backtest")


@dataclass
class BacktestResult:
    """Container for backtest output."""
    equity_curve: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    signals: List[dict] = field(default_factory=list)  # per-day signal log


def kelly_fraction(
    p_up: float,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
    cap: float = BT_KELLY_CAP,
) -> float:
    """
    Compute Kelly Criterion sizing.

    f* = (p * b - q) / b
    where p = P(win), q = 1-p, b = avg_win / avg_loss

    Capped at `cap` for safety.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    b = avg_win / avg_loss
    q = 1.0 - p_up
    f = (p_up * b - q) / b

    return max(0.0, min(f, cap))


def run_backtest(
    dates: pd.DatetimeIndex,
    actual_returns: np.ndarray,
    calibrated_probs: np.ndarray,
    predicted_1d_returns: Optional[np.ndarray] = None,
    initial_capital: float = BT_INITIAL_CAPITAL,
    signal_threshold: float = BT_SIGNAL_THRESHOLD,
    kelly_cap: float = BT_KELLY_CAP,
    commission_bps: float = BT_COMMISSION_BPS,
) -> BacktestResult:
    """
    Run a simple long/flat backtest based on calibrated direction probabilities.

    Args:
        dates: trading dates
        actual_returns: actual daily returns (decimal, not %)
        calibrated_probs: calibrated P(up) for each day
        predicted_1d_returns: optional predicted return magnitudes
        initial_capital: starting capital
        signal_threshold: P(up) above which we go long
        kelly_cap: maximum position fraction
        commission_bps: round-trip commission in basis points

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    n = len(dates)
    if n == 0:
        return BacktestResult(metrics={"error": "no data"})

    commission_rate = commission_bps / 10_000

    equity = initial_capital
    equity_curve = [equity]
    date_strs = [str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0])]
    trades = []
    signals = []

    position = 0.0  # fraction of capital invested
    in_trade = False
    trade_entry_equity = equity
    trade_entry_date = None

    # Running stats for Kelly sizing
    wins = []
    losses = []

    for i in range(n):
        p_up = float(calibrated_probs[i])
        actual_ret = float(actual_returns[i]) if np.isfinite(actual_returns[i]) else 0.0

        # Determine signal
        if p_up > signal_threshold:
            signal = "LONG"
        elif p_up < (1.0 - signal_threshold):
            signal = "EXIT"
        else:
            signal = "HOLD"

        # Compute Kelly fraction for position sizing
        avg_win = float(np.mean(wins)) if wins else 0.01
        avg_loss = float(np.mean(np.abs(losses))) if losses else 0.01
        k_frac = kelly_fraction(p_up, avg_win, avg_loss, cap=kelly_cap)

        # Execute signal
        prev_position = position
        if signal == "LONG" and position == 0:
            position = k_frac if k_frac > 0.01 else 0.05  # minimum allocation
            in_trade = True
            trade_entry_equity = equity
            trade_entry_date = dates[i]
            # Commission on entry
            equity -= equity * position * commission_rate

        elif signal == "EXIT" and position > 0:
            # Close position
            equity -= equity * position * commission_rate  # exit commission
            position = 0.0
            if in_trade:
                pnl = equity - trade_entry_equity
                pnl_pct = pnl / trade_entry_equity if trade_entry_equity > 0 else 0
                trades.append({
                    "entry_date": str(trade_entry_date.date()) if hasattr(trade_entry_date, "date") else str(trade_entry_date),
                    "exit_date": str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]),
                    "pnl": round(float(pnl), 2),
                    "pnl_pct": round(float(pnl_pct * 100), 4),
                    "entry_equity": round(float(trade_entry_equity), 2),
                    "exit_equity": round(float(equity), 2),
                })
                if pnl > 0:
                    wins.append(pnl_pct)
                else:
                    losses.append(pnl_pct)
                in_trade = False

        # Apply daily return to invested portion
        daily_pnl = equity * position * actual_ret
        equity += daily_pnl
        equity = max(equity, 0.01)  # prevent going below zero

        equity_curve.append(equity)
        date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
        date_strs.append(date_str)

        signals.append({
            "date": date_str,
            "p_up": round(p_up, 4),
            "signal": signal,
            "position": round(position, 4),
            "kelly": round(k_frac, 4),
            "equity": round(equity, 2),
            "daily_return": round(actual_ret * 100, 4),
        })

    # ── Compute performance metrics ──────────────────────────────────
    metrics = compute_backtest_metrics(
        np.array(equity_curve),
        trades,
        initial_capital,
    )

    # Buy & hold benchmark
    bh_equity = initial_capital
    bh_curve = [bh_equity]
    for i in range(n):
        ret = float(actual_returns[i]) if np.isfinite(actual_returns[i]) else 0.0
        bh_equity *= (1 + ret)
        bh_curve.append(bh_equity)

    metrics["benchmark_buy_hold_return"] = round(
        float((bh_equity - initial_capital) / initial_capital * 100), 4
    )

    return BacktestResult(
        equity_curve=equity_curve,
        dates=date_strs,
        trades=trades,
        metrics=metrics,
        signals=signals,
    )


def compute_backtest_metrics(
    equity_curve: np.ndarray,
    trades: List[dict],
    initial_capital: float,
) -> Dict:
    """Compute performance metrics from equity curve and trades."""
    n = len(equity_curve)
    if n < 2:
        return {"total_return_pct": 0.0}

    final = equity_curve[-1]
    total_return = (final - initial_capital) / initial_capital * 100

    # Daily returns from equity curve
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    trading_days = len(daily_returns)
    years = max(trading_days / 252, 0.01)

    # Annualised return
    ann_return = ((final / initial_capital) ** (1 / years) - 1) * 100

    # Volatility
    daily_vol = float(np.std(daily_returns)) if len(daily_returns) > 1 else 0
    ann_vol = daily_vol * np.sqrt(252) * 100

    # Sharpe Ratio (risk-free = 0)
    mean_daily = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0
    sharpe = (mean_daily / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0

    # Sortino Ratio (downside deviation)
    downside = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(downside)) if len(downside) > 1 else daily_vol
    sortino = (mean_daily / downside_vol * np.sqrt(252)) if downside_vol > 0 else 0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = float(np.min(drawdown)) * 100

    # Calmar Ratio
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.001 else 0

    # Win rate & profit factor from trades
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return_pct": round(float(total_return), 4),
        "annualised_return_pct": round(float(ann_return), 4),
        "annualised_volatility_pct": round(float(ann_vol), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "calmar_ratio": round(float(calmar), 4),
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": round(float(win_rate), 2),
        "profit_factor": round(float(profit_factor), 4),
        "gross_profit": round(float(gross_profit), 2),
        "gross_loss": round(float(gross_loss), 2),
        "trading_days": trading_days,
        "initial_capital": float(initial_capital),
        "final_equity": round(float(equity_curve[-1]), 2),
    }
