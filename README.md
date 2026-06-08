# Stock Direction Model (XGBoost + Logistic Regression)

A short-horizon trading model for around 70 large-cap US stocks. It predicts
near-term direction from daily price data, and there's a backtester that turns those
predictions into actual trades with ATR-based stops, a 50-day moving average filter,
conviction-drop exits, and earnings avoidance.

I built this to find out how much signal you can realistically pull out of daily
prices on the most liquid names, then spent a while stress-testing it to figure out
where the edge actually comes from. The honest answer turned out to be more
interesting than the model itself.

## The model

It's a soft-vote ensemble of XGBoost (weight 0.55) and an L1 logistic regression
(0.45). Each stock/day gets about 60 features:

- **Technical:** RSI, MACD, ROC, distance from various moving averages, Bollinger
  position, realised volatility, return stats, price slope, volume ratios, etc.
- **Macro / regime:** SPY momentum and trend, SPY realised vol, VIX level.
- **Analyst:** buy/sell consensus, price-target upside, EPS revision momentum.

Labels are ATR-based. A day is labelled "up" if the 5-day forward return clears
+1 ATR, "down" if it falls below -1 ATR, and the noisy middle is dropped so the model
only trains on decisive moves. The data is split by time (train / val / test) with a
6-month purge gap between train and validation so nothing leaks across the boundary.

## The honest result

On the held-out test set the ensemble lands at about **0.53 AUC**. That's only just
above a coin flip, which is roughly what you should expect for daily direction on big
liquid stocks. That part of the market is extremely efficient.

What I found more interesting was *why* it's only 0.53. The model's most important
features are nearly all macro (VIX, SPY trend), which means it isn't really picking
stocks at all. It's reading whether the overall market is risk-on and applying that
to everything. So the "edge" is mostly market timing in disguise.

I went down a long rabbit hole trying to fix that, which is the whole `research/`
folder. The summary: the technical features have basically no durable stock-vs-stock
signal at any horizon I tested, and the only thing that held up out of sample was
market timing, not stock selection. Full write-up in
[`research/FINDINGS.md`](research/FINDINGS.md).

## Layout

```
.
├── data.py          download prices, build features + labels, save train/val/test
├── XGB.py           grid-search XGBoost, train the LR, evaluate the ensemble
├── backtest.py      run the model as a trading strategy, log trades + daily P&L
├── research/        the deeper investigation into where the edge really is
│   └── FINDINGS.md  read this for the interesting part
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

python data.py        # downloads ~10y of data and builds the dataset (a few minutes)
python XGB.py         # trains and evaluates the ensemble
python backtest.py    # backtests the saved model
```

A couple of notes:

- Everything is pulled live from Yahoo Finance, so you need a connection and the
  numbers will drift slightly over time as the data updates.
- `XGB.py` trains on GPU (`device="cuda"`). If you don't have one, change that to
  `device="cpu"` in the model setup.
- The trained model files are included, so you can run `backtest.py` straight away
  without retraining.

## What I'd do next

The research points pretty firmly at market timing being the real opportunity. The
macro signal is stable across time; the stock-level signal isn't. If I picked this
back up, I'd stop trying to pick individual stocks and build a proper market-exposure
model around that instead. There's more detail on that, with the evidence, in the
findings doc.
