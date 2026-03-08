"""
ViziGenesis V2 — fred_data.py (DEPRECATED)
==========================================
This module is a backward-compatibility shim.
All functionality has moved to ``market_data.py``.

Any import from ``backend.v2.fred_data`` is transparently redirected.
"""
# Re-export everything that external callers expect
from backend.v2.market_data import (          # noqa: F401
    fetch_macro_data   as fetch_fred_macro,
    fetch_fred_macro,                          # alias already exists
    build_fomc_features,
    fetch_market_context,
    fetch_sector_commodity,
    fetch_nasdaq_close,
)
