# NVIDIA Next-CEO Price Forecaster

Entry for make.fun bounty `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every trading market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after a checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%) per market per minute. Scored by mean pinball loss;
a prize needs a loss below the best house entry (persistence baseline:
0.363589 on the practice cases).

## Method (`main.py`, Python 3.13, standard library only)

For each market, with `x` its last price and `h` the minutes ahead:

```
q_tau(h) = x + pull(h) + 0.5 * z_tau * 0.0025 * sqrt(h) + 0.5 * emp_tau(h)
```

- **Baseline half**: persistence with a random-walk spread (the house
  baseline's formula).
- **Empirical half**: `emp_tau(h)` is the tau-quantile of the market's own
  h-minute price changes, estimated from its 72-hour history with
  recency weights (6-hour half-life) at log-spaced anchor horizons
  (1, 2, 4, 8, 16, 32, 64, 120, 240 minutes), interpolated in log(h).
  This captures per-market volatility, the fact that prices usually do
  not move at all (inner quantiles sit near `x`), jumps (fat outer
  tails), and asymmetry.
- **Sum-to-one pull**: the 24 Yes prices are probabilities of mutually
  exclusive outcomes, so they should sum to about 1. If the sum `S` at
  the checkpoint is off 1, each market's centre is pulled toward its
  fair price `x / S`: `pull(h) = x * (1 - S) / S * sqrt(h / 240)`.
  Big markets carry most of the correction.

Quantiles are clamped to [0, 1], rounded to 6 decimals, and forced
non-decreasing. The program is deterministic (same case → same bytes)
and reads `$MAKEFUN_INPUT_DIR` (else `/input`), writing to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Local results (practice cases, actuals fetched from Polymarket's
public price history)

| case | this entry | house baseline |
|------|-----------:|---------------:|
| p00  |     0.4512 |         0.5361 |
| p01  |     0.3707 |         0.3804 |
| p02  |     0.3890 |         0.4084 |
| p03  |     0.3409 |         0.3686 |
| p04  |     0.2849 |         0.2988 |
| p05  |     0.1905 |         0.2382 |
| **overall** | **0.3301** | **0.3636** |

Mean pinball loss 0.330134 vs the baseline's 0.363589 (-9.2%).
Runtime ~0.3–0.5 s per case (limit 8 s). Practice scores are no guide
to the final — the model was kept simple and structural (per-market
empirical quantiles + the sum-to-one constraint) rather than tuned to
the six practice cases.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=../kit/polymarket-nvidia-ceo-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see `LICENSE`).
