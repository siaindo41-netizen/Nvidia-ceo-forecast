#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute,
4 hours ahead, as five quantiles per market per minute.

Method (v2): a blend of the house persistence baseline with the market's own
recent behaviour, plus a cross-market correction.

1. Baseline (weight 0.5): the price stays where it was, spread widening
   like a random walk, q = x + z_tau * 0.0025 * sqrt(h).

2. Empirical (weight 0.5): for each market, the quantiles of its own
   h-minute price changes, estimated from its 72-hour history with
   recency weights (6-hour half-life) at log-spaced anchor horizons
   (1..240 minutes) and interpolated in log(h). This captures, per market,
   that prices usually do not move at all (inner quantiles near x), that
   jumps happen (fat outer tails), that moves are asymmetric, and that
   each market has its own volatility.

3. Sum-to-one correction: the 24 Yes prices are probabilities of mutually
   exclusive outcomes, so they should sum to about 1. If the sum S at the
   checkpoint is off 1, each market's centre is pulled toward its fair
   price x / S, by x * (1 - S) / S * sqrt(h / 240) (full correction by the
   end of the horizon, faster early). Big markets carry most of the
   correction; tiny markets barely move.

Standard library only. Deterministic: the same case always gives the same
bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output). Runs in well under a second per case on one CPU.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Anchor horizons (minutes) for the empirical change distribution.
ANCHORS = [1, 2, 4, 8, 16, 32, 64, 120, 240]
HALF_LIFE_MIN = 360.0   # recency half-life for the empirical quantiles
EMP_WEIGHT = 0.5        # weight of the empirical quantiles vs the baseline
BASE_SIGMA = 0.0025     # the house baseline's spread per sqrt-minute
MIN_SAMPLES = 20        # an anchor needs at least this many windows


def _wquantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs."""
    pairs.sort()
    total = 0.0
    for _, w in pairs:
        total += w
    if total <= 0.0:
        return 0.0
    threshold = tau * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= threshold:
            return v
    return pairs[-1][0]


def empirical_quantiles(prices):
    """{anchor horizon: [q05, q25, q50, q75, q95] of the h-minute change}."""
    n = len(prices)
    out = {}
    for h in ANCHORS:
        if n - h < MIN_SAMPLES:
            continue
        samples = []
        for i in range(n - h):
            age = float(n - h - i)
            w = 0.5 ** (age / HALF_LIFE_MIN)
            samples.append((prices[i + h] - prices[i], w))
        out[h] = [_wquantile(samples, tau) for tau in TAUS]
    return out


def _interp(emp, h):
    """Empirical quantiles at horizon h, log-linear between anchors."""
    hs = sorted(emp)
    if not hs:
        return [0.0] * 5
    if h <= hs[0]:
        return emp[hs[0]]
    if h >= hs[-1]:
        return emp[hs[-1]]
    for a, b in zip(hs, hs[1:]):
        if a <= h <= b:
            f = (math.log(h) - math.log(a)) / (math.log(b) - math.log(a))
            return [emp[a][i] + f * (emp[b][i] - emp[a][i])
                    for i in range(5)]
    return emp[hs[-1]]


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def forecast_market(prices, horizon, total_sum):
    """240 rows of five quantiles for one market."""
    x = prices[-1]
    emp = empirical_quantiles(prices)
    has_emp = bool(emp)
    # Proportional pull toward the fair price x / total_sum.
    pull = 0.0
    if total_sum > 0.0:
        pull = x * (1.0 - total_sum) / total_sum
    rows = []
    for h in range(1, horizon + 1):
        centre = x + pull * math.sqrt(h / 240.0)
        spread = math.sqrt(h)
        if has_emp:
            eq = _interp(emp, h)
        else:
            eq = [0.0] * 5
        row = []
        prev = None
        # No empirical estimate (history too short): fall back to the full
        # baseline Gaussian instead of a halved one.
        w_emp = EMP_WEIGHT if has_emp else 0.0
        for ti in range(5):
            g = Z[ti] * BASE_SIGMA * spread
            qv = centre + (1.0 - w_emp) * g + w_emp * eq[ti]
            q = price6(qv)
            if prev is not None and q < prev:
                q = prev
            row.append(q)
            prev = q
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        markets = case["markets"]
        total_sum = 0.0
        for m in markets:
            if not m["prices"]:
                raise ValueError("market %s has no prices" % m["id"])
            total_sum += m["prices"][-1]
        forecasts = []
        for m in markets:
            forecasts.append({"id": m["id"],
                              "q": forecast_market(m["prices"], horizon,
                                                   total_sum)})
        out_name = name[:-3] + ".out"
        with open(os.path.join(OUTPUT_DIR, out_name), "w",
                  encoding="utf-8") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
