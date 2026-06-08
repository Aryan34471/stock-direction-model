# Research notes: where is the actual edge?

The baseline model gets about 0.53 test AUC. This folder is me trying to answer two
questions properly: where is that (weak) signal actually coming from, and can I make
it better?

Short version: the baseline's edge is almost entirely market timing dressed up as
stock-picking, the technical features have no durable stock-vs-stock signal at any
horizon I tested, and the one thing that does hold up out of sample (a contrarian
market-timing signal) still doesn't beat buy-and-hold once you account for risk.
Worth knowing before building anything real on top of it.

## What I tried

| Experiment | Test result | Takeaway |
|---|---|---|
| Baseline (absolute direction + macro) | AUC 0.53 | Weak, and it's market timing not stock-picking |
| Cross-sectional, 5-day horizon | AUC 0.499 | No signal at all |
| IC sweep across 5/10/21/63 days | mean \|IC\| 0.016 -> 0.053 | Signal exists, but only at ~quarter horizon, and it's regime-dependent |
| Cross-sectional, 63-day horizon | AUC 0.503 | Strong in-sample, gone out of sample |
| Long/short factor backtest | +12.8%/yr in-sample, ~0% OOS | The in-sample alpha is an illusion |
| Walk-forward (rolling retrain) | +1.9%/yr OOS | Helps a little, still not tradeable |
| Market-timing check | corr +0.25 -> +0.46, sign-stable | The real signal: contrarian market timing |
| Contrarian rule vs buy-and-hold | Sharpe 0.83 vs 1.03 | Real signal, but naive trading is worse risk-adjusted |

## The macro problem

The first thing I noticed about the baseline is that its most important features are
all macro: VIX level, SPY versus its 200-day average, SPY momentum. Eight of the top
ten. That means on any given day every stock gets basically the same signal, so the
model isn't choosing between stocks, it's deciding whether the whole market is
risk-on.

To test that, I reframed the problem. Instead of "does this stock go up," predict
"does this stock beat the rest of the universe over the next N days," and rank every
feature cross-sectionally each day so the macro stuff (identical across stocks) drops
out on its own.

At a 5-day horizon this killed the signal completely (AUC 0.499). Which confirmed the
suspicion: strip out market timing and there's nothing left.

## Where the signal actually is

Before retraining blindly I ran an information-coefficient sweep (`run_ic_diag.py`),
measuring the rank correlation between each feature and forward returns across several
horizons. The signal clearly grows with horizon:

```
test-period mean |IC|:   5d = 0.016    10d = 0.022    21d = 0.020    63d = 0.053
```

So there is cross-sectional signal, but it only really shows up around a quarter, not
a week. The catch is that the features carrying it flip sign between the training era
and the test era. In 2015-2023 it's volatility (high-vol stocks outperform); in
2024-2025 it's reversal (recent winners underperform). Same features, opposite
relationship.

## Non-stationarity, made visible

I retrained at the 63-day horizon and the validation numbers looked great (0.578
balanced accuracy). Then the long/short factor backtest (`backtest_xsec.py`) showed
what was really going on. Sorting all stocks into quintiles by model score and
measuring the realised forward returns:

```
full sample:   2.48 -> 2.77 -> 3.46 -> 4.27 -> 5.60     clean, monotonic, +12.8%/yr L/S
test period:   2.79 -> 0.78 -> 1.57 -> 2.27 -> 2.90     flat and noisy, ~0%/yr
```

The model ranks beautifully where it trained and randomly where it didn't. That
+12.8% in-sample number is exactly the kind of thing that looks like a winning
strategy and means nothing.

Walk-forward retraining (`walk_forward.py`, retrain monthly on a rolling 3-year
window, with a purge so there's no leakage) recovered a sliver, +1.9%/yr, but that's
well inside the noise for the number of independent observations you actually have.
Not tradeable after costs.

## The one thing that held up

The cross-sectional signal flips sign between eras. The market-timing signal doesn't.
I checked whether the macro features predict the *market's* forward return rather than
individual stocks (`market_timing_diag.py`), and the relationships are stable across
both periods and economically sensible:

```
                  full sample (63d)    test period (63d)
vix_level              +0.25                +0.46
spy_vol_20             +0.19                +0.45
spy_vs_sma200          -0.15                -0.34
spy_rsi_14             -0.19                -0.19
```

High VIX, high volatility, oversold, below trend, all predict higher forward returns.
That's contrarian mean reversion: fear precedes bounces. The signs agree across
periods, which is exactly what the stock-selection signal failed to do.

The obvious caveat is that these are overlapping windows, so the 63-day test numbers
rest on only a handful of independent observations. The part that matters is the sign
stability across the full sample and the consistency across horizons, not the exact
magnitude.

## But you can't just trade it

I built a simple, non-fitted contrarian rule (`contrarian_timing.py`): lever up when
the signal says fearful, down when euphoric, with the signs fixed from economic priors
and no threshold tuning, then compared it to buy-and-hold:

```
                  Sharpe    max drawdown
buy & hold         1.03        -19%
contrarian         0.83        -33%        (test period)
```

It gets similar or higher raw returns but worse risk-adjusted, because levering up on
high VIX means levering into crashes. The signal is real; turning it into an edge
needs proper risk-aware sizing (volatility targeting, which pulls the opposite way),
and even then the net edge is probably modest.

## Takeaway

The same lesson showed up three separate times here: a real statistical signal is not
a tradeable edge. The in-sample long/short, the raw correlations, and the contrarian
rule each looked good on one axis and fell apart on another. The only way to catch it
was to always check out of sample *and* risk-adjusted.

If I built on this, I'd drop the stock-selection angle and go after market timing
properly, since that's where the durable signal lives. The scripts in this folder are
the tooling for doing that kind of evaluation honestly:

- `run_ic_diag.py` — does signal exist, and at what horizon?
- `build_from_panel.py` — rebuild the dataset at any horizon from the cached data
- `backtest_xsec.py` — market-neutral long/short factor backtest
- `walk_forward.py` — leak-free rolling-retrain test
- `market_timing_diag.py` — does macro predict the market itself?
- `contrarian_timing.py` — can you actually trade the market-timing signal?
