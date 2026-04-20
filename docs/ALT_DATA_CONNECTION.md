# Alt-Data Equity Signals Connection

This repo can now consume retail-attention factors exported by
[`alt-data-equity-signals`](https://github.com/kartikmunjal/alt-data-equity-signals).

## Why This Connection Matters

The securities-lending repo studies short interest, borrow pressure, and squeeze
risk. Retail attention is a natural second crowding channel:

```text
short crowding:  short interest, days-to-cover, borrow stress
retail crowding: WSB mentions, sentiment, attention shocks
```

The connected workflow tests whether social attention adds explanatory power to
short-interest and borrow-cost signals.

## Data Contract

The alt-data repo exports:

```text
factor_panels/
├── WSB_MENTION_Z.parquet
├── WSB_SENTIMENT_Z.parquet
└── WSB_ATTENTION_SHOCK_Z.parquet
```

`src/securities_lending/features/retail_attention.py` converts those `date x
ticker` panels into the long feature format used here:

```text
date, symbol, wsb_mention_z, wsb_sentiment_z, wsb_attention_shock_z
```

## Run Analysis With Retail Attention

```bash
python scripts/run_analysis.py \
  --features data/processed/features.parquet \
  --alt-factor-dir ../alt-data-equity-signals/results/wsb_retail_attention/factor_panels \
  --output-dir data/results/with_retail_attention
```

The analysis automatically includes:

- `wsb_mention_z`
- `wsb_sentiment_z`
- `wsb_attention_shock_z`
- `borrow_stress_x_wsb_attention`
- `dtc_x_wsb_attention`
- `short_pressure_x_wsb_sentiment`

These are evaluated in the same IC, portfolio-sort, and Fama-MacBeth framework
as the original lending signals.

## Squeeze Model Extension

The squeeze detector now treats WSB retail-attention columns as optional model
features. If they exist in the feature frame, they are used; if not, the original
short-interest-only workflow still runs.

The strongest research question is:

> Are crowded shorts with abnormal WSB attention more likely to produce positive
> forward-return dislocations or short-squeeze events?

## Ownership Boundary

`alt-data-equity-signals` remains the source of truth for messy social-data
ingestion, ticker extraction, sentiment scoring, and alt-data panel generation.
This repo consumes finished panels and studies how they interact with securities
lending mechanics.
