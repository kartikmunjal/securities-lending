"""
securities-lending
==================
Short interest and borrow rate analysis pipeline.

Modules
-------
ingestion   — Download FINRA Reg SHO, short interest, and price data.
features    — Compute short metrics, microstructure proxies, and borrow rate proxy.
analysis    — IC analysis, portfolio sorts, and Fama-MacBeth regression.
models      — Short-squeeze detection model with walk-forward evaluation.
viz         — Plotting utilities (tear sheets, decay curves, SHAP beeswarms).
utils       — Calendar helpers, winsorization, parallel execution.
"""

__version__ = "0.1.0"
