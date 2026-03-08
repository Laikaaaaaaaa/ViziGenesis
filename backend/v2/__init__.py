"""
ViziGenesis V2 — Institutional-Grade Quant Forecasting System
=============================================================
Multi-horizon, multi-stock, regime-aware forecasting & trading pipeline.
Understands macro/Fed policy and market psychology.

Modules:
  config       – central configuration & hyperparameters
  market_data  – multi-source market data engine (60+ tickers, World Bank, etc.)
  sentiment    – news & social sentiment signals
  features     – unified feature engineering (160+ features)
  regime       – regime detection (bull / bear / sideways)
  panel_data   – panel data loader for cross-stock training
  model        – hybrid deep-learning architecture (TFT encoder + heads)
  tree_models  – LightGBM / XGBoost tabular heads
  meta_router  – learned meta-router for model weighting
  calibration  – Isotonic Regression + Platt Scaling
  walk_forward – walk-forward cross-validation engine
  backtest     – full backtest with Kelly sizing & stress tests
"""
__version__ = "2.0.0"
