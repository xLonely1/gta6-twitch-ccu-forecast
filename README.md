# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xLonely1/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xLonely1/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour 2..24 of its release day (2026-11-19 UTC), as five quantiles per hour. One file, `main.py`, Python 3.13 standard library only, no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kkhbctt2&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## Model

Each comparable is a past big launch with hourly Twitch viewers. Its launch-day
curve is normalised to its own hour-1 print (log-ratios `ln(viewers_h /
viewers_1)`), and the forecast for hour `h` is hour-1's print times `exp()` of
the distribution of those log-ratios across comparables — median for q50,
distribution quantiles for q05/q25/q75/q95. The pinball loss on the log scale
is a proper scoring rule for exactly these quantiles.

Comparables are filtered and weighted from the case file itself:

- **Complete launch day only** (all 24 hours, every hour > 0). The GTA VI
  category's own pre-release history has no launch day and drops out.
- **Hour 1 must be a real launch hour**: hour-1 viewers ≥ 5% of the
  comparable's launch-day peak. Simultaneous-global-unlock games sit near zero
  at 00:00 UTC and explode 100–2000× later; their ratios describe a different
  event, not a rolling midnight launch.
- **No early-access-contaminated launches** (flagged `Tier:
  early-access-contaminated` in the case's own research notes): their launch
  day is flat because the audience arrived days earlier.
- **Similarity weights**: console-only + rolling-midnight unlock ×3, either ×2,
  else ×1. GTA VI is console-only, midnight local, rolling.
- **Scale kernel**: a Gaussian kernel (bandwidth 0.75 log units) on
  ln(hour-1 viewers) around the target's own hour-1. Across comparables the
  day's growth is strongly tied to how big hour 1 already is (corr ≈ −0.8 for
  the evening hours): a small first hour means the big unlock waves are still
  ahead; a huge first hour means the wave already arrived.

If no comparable survives, the program falls back to the house persistence
baseline rather than crashing. See `main.py`'s docstring for the full design.

## Backtest (practice cases, answers from SullyGnome hourly averages)

| case | loss | typical miss | 90% coverage |
| ---- | ---- | ------------ | ------------ |
| p00 Marvel's Wolverine | 0.355 | 79.7% | 0.957 |
| p01 Red Dead Redemption 2 | 0.138 | 22.2% | 0.913 |
| p02 God of War Ragnarök | 0.129 | 17.1% | 0.957 |
| p03 Marvel's Spider-Man 2 | 0.191 | 30.9% | 0.783 |
| **mean** | **0.203** | **37.5%** | **0.902** |

House persistence baseline on the same cases: mean loss **0.387**, typical miss
82.8%, coverage 0.783. Runtime ~0.07 s per case; two runs byte-identical.
