"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Twitch viewers of Grand Theft Auto VI in each hour 2..24 of its
release day (UTC), as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), from the one
case file it is given. Reads every <id>.in in the input directory and writes
<id>.out to the output directory:
    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, hours 2..24.

How the grader runs it: `python3 -P -s /bot/main.py`, no network, deterministic,
<= 8 seconds per case. Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output.

MODEL (everything below is computed from the case file at run time; nothing is
memorised from the practice cases and no number is tuned per case):

Each comparable is a past big launch with hourly Twitch viewers from 14 days
before its release day to 3 days after it. For a comparable, take its launch-day
curve and normalise it to its own hour-1 print: the log-ratio
    lr_h = ln(viewers_h / viewers_1),  h = 2..24.
The forecast for hour h is hour-1's print times exp() of the distribution of
lr_h across comparables: the median for q50, the distribution's quantiles for
q05/q25/q75/q95. That is the honest conditional distribution of "how a launch
day grows after its first hour", and the pinball loss on the log scale is a
proper scoring rule for exactly these quantiles.

Not every comparable belongs in that distribution. Three filters, all read off
the case file:
  1. A comparable needs a complete launch day (all 24 hours, every hour > 0
     viewers). The GTA VI category's own pre-release history has no launch day
     and drops out here; so does any comparable with missing hours.
  2. Hour 1 must already be a launch hour: hour-1 viewers at least 5% of the
     comparable's launch-day peak. Simultaneous-global-unlock games (PC unlock
     at 18:00Z, say) sit near zero at 00:00 UTC and then explode 100-2000x;
     their ratios describe a different event (a later unlock), not a rolling
     midnight launch. The 5% cut sits in a wide empty gap (0.011 to 0.082 in
     the practice cases), so it is not a knife-edge.
  3. Early-access-contaminated launches are out: the case's own research notes
     flag them ("Tier: early-access-contaminated" in `unlock`). Their launch
     day is flat because the audience arrived days earlier; GTA VI has no
     early access.

Two weights, both from fields actually in the case:
  - Similarity: console-only AND rolling-midnight unlock x3 (GTA VI is
    console-only, midnight local, rolling); console-only OR rolling x2; else x1.
    "Console-only" = `platforms` names no PC; "rolling" = `unlock` says
    "midnight local".
  - Scale: a Gaussian kernel on ln(hour-1 viewers) around the target's own
    ln(hour-1), bandwidth 0.75 log units. Across the comparables, the day's
    growth is strongly tied to how big hour 1 already is (corr(log v1,
    log evening ratio) about -0.8): a small first hour means the big unlock
    waves are still ahead and the day grows steeply; a huge first hour means
    the wave already arrived and the day stays flat. The kernel lets games
    that started at a similar scale speak loudest, and it extrapolates sanely:
    a first hour beyond every comparable just concentrates weight on the
    biggest launches (the flattest curves).

Quantiles are interpolated weighted quantiles of the log-ratios, so q05 <= q25
<= q50 <= q75 <= q95 by construction; they are then rounded to 2 decimals
half up (the starter's rounding), clamped to (0, 1e9] and forced
non-decreasing as a safety net. One guardrail: the implied log-SD of an
hour's quantiles never falls below half the persistence baseline's
random-walk spread (0.15 * sqrt(h-1)) — it almost never binds, but it stops
the intervals collapsing when hour 1 is far beyond every comparable and the
kernel rests on two or three giants.

If no comparable survives the filters (it should not happen: the final case
carries 42), the program falls back to the house persistence baseline
(q = p1 * exp(z_tau * 0.15 * sqrt(h-1))) rather than crashing.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the persistence baseline's).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The persistence baseline's spread, in log units per square-root hour.
SIGMA = 0.15

# --- model constants (fixed by design, not tuned per case) -------------------
# A comparable's hour 1 counts as a launch hour when it is at least this
# fraction of its own launch-day peak.
MIN_HOUR1_SHARE_OF_PEAK = 0.05
# Similarity weights: console-only + rolling / either / neither.
WEIGHT_BOTH = 3.0
WEIGHT_EITHER = 2.0
WEIGHT_NEITHER = 1.0
# Gaussian kernel bandwidth on ln(hour-1 viewers) around the target's own.
KERNEL_BANDWIDTH = 0.75
# Spread floor: the implied log-SD of an hour's quantiles, (ln q95 - ln q05) /
# (2 * 1.645), is never allowed below this fraction of the persistence
# baseline's random-walk spread (0.15 * sqrt(h-1)). It almost never binds on
# ordinary cases (the backtest moves by +0.0001), but it stops the intervals
# collapsing when hour 1 is far beyond every comparable and the kernel rests
# on two or three giants: an unprecedented launch deserves honest uncertainty.
SPREAD_FLOOR_FRACTION = 0.5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0). Same helper as the starter.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def launch_hours(comp: dict) -> list[int] | None:
    """A comparable's launch-day hourly viewers, hours 1..24, or None.

    None when any of the 24 hours is missing from `hourly` (the GTA VI
    category's own pre-release history has no launch day at all).
    """
    hours = {row["from"]: row["viewers"] for row in comp.get("hourly", [])}
    try:
        base = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    out = []
    for h in range(24):
        value = hours.get((base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z"))
        if value is None:
            return None
        out.append(value)
    return out


def is_console_only(comp: dict) -> bool:
    return "pc" not in (comp.get("platforms") or "").lower()


def is_rolling(comp: dict) -> bool:
    return "midnight local" in (comp.get("unlock") or "").lower()


def is_early_access(comp: dict) -> bool:
    return "early-access-contaminated" in (comp.get("unlock") or "")


def is_not_a_launch(comp: dict) -> bool:
    return (comp.get("unlock") or "").startswith("Not a launch")


def collect(case: dict) -> list[tuple[list[float], float, float]]:
    """Usable comparables as (log-ratios for h=2..24, weight, hour-1 viewers).

    The weight here is the similarity weight only; the scale kernel is applied
    in forecast_case, where the target's hour-1 print is known.
    """
    rows: list[tuple[list[float], float, float]] = []
    for comp in case.get("comparables", []):
        if is_not_a_launch(comp) or is_early_access(comp):
            continue
        viewers = launch_hours(comp)
        if viewers is None or any(v is None or v <= 0 for v in viewers):
            continue
        first, peak = viewers[0], max(viewers)
        if first / peak < MIN_HOUR1_SHARE_OF_PEAK:
            continue
        ratios = [math.log(viewers[h] / first) for h in range(1, 24)]
        if is_console_only(comp) and is_rolling(comp):
            weight = WEIGHT_BOTH
        elif is_console_only(comp) or is_rolling(comp):
            weight = WEIGHT_EITHER
        else:
            weight = WEIGHT_NEITHER
        rows.append((ratios, weight, float(first)))
    if not rows:
        # Nothing survived: fall back to every comparable with a complete,
        # positive launch day, unfiltered, rather than failing the case.
        for comp in case.get("comparables", []):
            viewers = launch_hours(comp)
            if viewers is None or any(v is None or v <= 0 for v in viewers):
                continue
            ratios = [math.log(viewers[h] / viewers[0]) for h in range(1, 24)]
            rows.append((ratios, 1.0, float(viewers[0])))
    return rows


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Interpolated weighted quantile of [(value, weight)] at level tau."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    cumulative = 0.0
    prev_value, prev_cum = ordered[0][0], 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            if weight <= 0:
                return value
            frac = max(0.0, min(1.0, (target - prev_cum) / weight))
            return prev_value + frac * (value - prev_value)
        prev_value, prev_cum = value, cumulative
    return ordered[-1][0]


def baseline_quantiles(p1: float, hour: int) -> list[float | int]:
    """The house persistence baseline's five quantiles for one hour."""
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case: dict) -> dict:
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    rows = collect(case)
    if not rows:
        # No comparable at all: the persistence baseline, valid if unambitious.
        return {
            "hours": [
                {"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
                for row in case["forecast"]
            ]
        }
    log_p1 = math.log(p1)
    weighted = [
        (
            ratios,
            weight
            * math.exp(-0.5 * ((math.log(hour1) - log_p1) / KERNEL_BANDWIDTH) ** 2),
        )
        for ratios, weight, hour1 in rows
    ]
    hours = []
    for index, row in enumerate(case["forecast"]):
        pairs = [(ratios[index], weight) for ratios, weight in weighted]
        log_quantiles = [weighted_quantile(pairs, tau) for tau in LEVELS]
        median = log_quantiles[2]
        spread = (log_quantiles[4] - log_quantiles[0]) / (2 * 1.645)
        floor = SPREAD_FLOOR_FRACTION * SIGMA * math.sqrt(row["hour"] - 1)
        if 0 < spread < floor:
            widen = floor / spread
            log_quantiles = [median + (x - median) * widen for x in log_quantiles]
        quantiles = [round2(p1 * math.exp(x)) for x in log_quantiles]
        quantiles = [max(1, min(1_000_000_000, q)) for q in quantiles]
        for i in range(1, 5):  # safety net: quantiles never decrease
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": row["hour"], "q": quantiles})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
