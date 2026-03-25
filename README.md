# securities-lending

**Short Interest and Borrow Rate Analysis Pipeline**

A research-grade pipeline for studying the relationship between short interest, securities lending costs, market microstructure, and subsequent cross-sectional equity returns.

Built at the intersection of **securities lending mechanics**, **market microstructure**, and **systematic signal research** — the domain focus of Voleon's QTS team.

---

## Overview

Short sellers must borrow shares before selling them short.  The cost of that borrow (the *borrow rate* or *securities lending fee*) is set by supply-demand equilibrium in the lending market and can range from near-zero for large-cap liquid stocks to hundreds of basis points per year for hard-to-borrow (HTB) names.

This creates a rich signal landscape:

| Signal | Mechanism | Horizon |
|--------|-----------|---------|
| **Short Volume Ratio (SVR)** | Daily flow of informed short sellers | 1–5 days |
| **Short Interest % Float** | Aggregate directional bet against a stock | 5–21 days |
| **Days-to-Cover (DTC)** | Crowdedness of the short position | 5–21 days |
| **Borrow Rate Proxy** | Cost of maintaining the position → only informed sellers persist | 5–21 days |
| **Borrow Stress** | Supply-demand imbalance in lending market | 1–10 days |
| **Squeeze Setup Score** | Composite risk of forced short covering | 1–5 days |

---

## Data Sources

| Source | Data | Frequency | Access |
|--------|------|-----------|--------|
| [FINRA Reg SHO](https://cdn.finra.org/equity/regsho/daily/) | Short sale volume by ticker | Daily | Free |
| [FINRA Short Interest](https://cdn.finra.org/equity/regsho/biweekly/) | Aggregate short positions | Bi-monthly | Free |
| [yfinance](https://github.com/ranaroussi/yfinance) | Adjusted OHLCV, float shares | Daily | Free |
| **Borrow rate proxy** | Estimated from utilisation model | Daily | Derived |
| EquiLend / S3 Partners *(not included)* | Actual transaction borrow fees | Daily | Subscription |
| IBKR Securities Lending Dashboard *(stub included)* | Indicative borrow rates | Snapshot | Subscription |

> **On borrow rate data**: Actual transaction borrow rates are proprietary (S3 Partners, DataLend, IBKR).  This pipeline implements a utilisation-based proxy calibrated to the empirical relationship between short supply scarcity and lending fee levels documented in D'Avolio (2002).  The proxy is clearly labeled as an estimate throughout the codebase.

---

## Architecture

```
securities-lending/
├── src/securities_lending/
│   ├── ingestion/          # FINRA + yfinance data fetchers with caching + retry
│   ├── features/           # SVR, SI%, DTC, borrow proxy, microstructure
│   ├── analysis/           # IC analysis, portfolio sorts, Fama-MacBeth
│   ├── models/             # Squeeze detector (HGBC + walk-forward eval)
│   └── viz/                # Tear sheets, decay curves, SHAP plots
├── scripts/
│   ├── fetch_data.py       # Download all raw data
│   ├── build_features.py   # Compute feature panel
│   ├── run_analysis.py     # IC + portfolio sorts + FM regression
│   └── run_squeeze_model.py# Train + evaluate squeeze detector
├── notebooks/
│   ├── 01_data_pipeline.ipynb           # Data quality validation
│   ├── 02_short_interest_signals.ipynb  # Signal IC and portfolio sorts
│   ├── 03_borrow_rate_factor_analysis.ipynb  # Borrow proxy + FM regression
│   └── 04_squeeze_detection.ipynb       # Squeeze model + watchlist
├── configs/
│   ├── universe.yaml       # Ticker list, date ranges, data paths
│   └── pipeline.yaml       # Analysis parameters (IC horizons, TC scenarios, etc.)
└── tests/                  # Unit tests for core logic (run without real data)
```

---

## Installation

```bash
git clone https://github.com/kartikmunjal/securities-lending.git
cd securities-lending
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# Download data (FINRA + prices, ~2 years)
make fetch

# Build feature panel
make features

# Run signal analysis (IC, portfolio sorts, Fama-MacBeth)
make analyze

# Train and evaluate squeeze detector
make squeeze

# Or run everything end-to-end
make all
```

Results are saved to `data/results/`:
- `ic_summary.csv` — IC table with BH-corrected p-values
- `ic_tearsheet_*.png` — per-signal IC time-series, distribution, decay
- `quintile_sort_*.png` — Q1–Q5 return bar charts + net-of-cost sensitivity
- `fm_coefficients.png` — FM coefficients with Newey-West confidence intervals
- `squeeze/wf_metrics.png` — Walk-forward ROC-AUC, PR-AUC, Precision@10%

---

## Signal Research Methodology

### Information Coefficient (IC)

The IC measures the cross-sectional Spearman rank correlation between a signal and subsequent returns:

```
IC_t = SpearmanCorr_i[ signal_{i,t},  r_{i, t→t+h} ]
```

We use Spearman rather than Pearson for robustness to return outliers (short squeezes, earnings gaps).  Multiple testing across signals and horizons is controlled with **Benjamini-Hochberg FDR correction** at q=0.05.

### Portfolio Sorts

Stocks are sorted into quintiles on each signal and equal-weighted portfolios are held for 5 days.  We report:
- Gross L/S spread (Q5 − Q1)
- Net spread after transaction costs at 5, 10, 20 bps one-way
- Net spread after borrow cost (estimated from proxy for the short leg)
- Monotonicity score (fraction of adjacent quantile pairs where return increases)

### Fama-MacBeth Regression

Cross-sectional OLS at each date, averaged over time (Fama & MacBeth 1973).  Standard errors use **Newey-West correction** (4 lags) to account for autocorrelation from slow-moving SI data.  All signals are cross-sectionally standardised before regression so coefficients are directly comparable.

We test incremental contribution by comparing R² from M0 (controls only) vs M1 (controls + signal), using a paired t-test on ΔR²_t.

### Short Squeeze Detector

Labels: 5-day return > 15% + DTC > 5 + volume > 2× 20d avg (~1-3% event rate).

Model: `HistGradientBoostingClassifier` wrapped in `CalibratedClassifierCV` (isotonic) for well-calibrated probabilities.

Evaluation: monthly walk-forward OOS (252d train / 63d test / 21d step).  Primary metric: **PR-AUC** (more informative than ROC-AUC for rare events).

---

## Borrow Rate Proxy

The proxy maps estimated lending utilisation to an annualised borrow fee using a piecewise schedule calibrated to industry convention:

| Utilisation | Tier | Proxy Rate |
|-------------|------|------------|
| < 50% | Easy-to-borrow (ETB) | ~25–50 bps |
| 50–70% | Moderate | ~50–150 bps |
| 70–90% | Hard-to-borrow (HTB) | ~150–800 bps |
| > 90% | Special / "squeezable" | > 800 bps |

The nonlinear relationship above 80% utilisation is the key feature: small increases in short demand cause large increases in borrow cost, which is the mechanism underlying short squeeze dynamics (D'Avolio 2002, Drechsler & Drechsler 2016).

**Proxy limitations** (documented in `src/securities_lending/features/borrow_proxy.py`):
1. Lendable supply estimated as `float × 0.20`; actual lendable supply varies by institutional holding concentration.
2. Does not capture supply-side shocks (large lender recalls, ETF creation/redemption).
3. Not differentiated by prime broker — rates vary across brokers for the same security.

---

## Key References

| Paper | Relevance |
|-------|-----------|
| D'Avolio (2002) *The Market for Borrowing Stock* — JFE | Foundational empirical work on borrow cost structure |
| Drechsler & Drechsler (2016) *The Shorting Premium* | Borrow costs and cross-sectional return premia |
| Asquith, Pathak & Ritter (2005) *Short Interest, Institutional Ownership, and Stock Returns* — JFE | SI% as a return predictor |
| Engelberg, Reed & Ringgenberg (2012) *How are Shorts Informed?* — JFE | SVR as information signal |
| Diether, Malloy & Scherbina (2002) *Differences of Opinion and the Cross Section of Stock Returns* — JF | Disagreement and short interest |
| Fama & MacBeth (1973) *Risk, Return, and Equilibrium* — JPE | Regression methodology |
| Newey & West (1987) *A Positive Semi-Definite HAC Covariance Matrix* — Econometrica | Standard errors for FM regression |
| Amihud (2002) *Illiquidity and Stock Returns* — JFMR | Microstructure illiquidity measure |

---

## Design Decisions

**Why Spearman IC, not Pearson?** Spearman is robust to the fat-tailed return distribution of a universe that includes meme stocks and squeeze events.  Pearson IC in a universe containing GME Jan 2021 would be dominated by a single observation.

**Why multiple-testing correction?** Testing 6 signals × 4 horizons = 24 hypotheses with no correction at p<0.05 would expect 1.2 false positives by chance.  BH FDR at q=0.05 controls the expected fraction of false discoveries.

**Why walk-forward for the squeeze model?** A single train/test split leaks information about the entire history into the training set.  Walk-forward mirrors production retraining frequency and gives an honest OOS estimate.

**Why PR-AUC over ROC-AUC for squeeze detection?** With ~2% event rate, a model predicting all zeros achieves ROC-AUC ≈ 0.5 and accuracy = 98%.  PR-AUC directly measures performance on the minority class.

**Why not vol-weight the portfolios?** Equal-weighting is used for IC and sort diagnostics to isolate the signal from the well-known size/liquidity interaction.  Production implementation would use risk-parity or dollar-neutral construction.

---

## Tests

```bash
make test
```

The test suite uses synthetic data only — no real FINRA or price data required.

```
tests/
├── conftest.py                          # Synthetic data fixtures
├── test_features/
│   ├── test_short_metrics.py            # SVR z-score, SI%, squeeze setup
│   └── test_borrow_proxy.py             # Rate schedule monotonicity, bounds
└── test_analysis/
    ├── test_ic_analysis.py              # IC, ICIR, BH correction
    └── test_portfolio_sorts.py          # Quintile construction, cost adjustment
```

---

## License

MIT
