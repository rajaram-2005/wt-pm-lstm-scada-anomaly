# The fidelity ladder

A predictive-maintenance result is only as meaningful as the data it was
produced on. "F1 = 0.69 on SCADA anomaly detection" means nothing on its own:
the same number can describe a simulation with exact labels and a field trial
with six confirmed failures. This document fixes a shared vocabulary so that
metrics from the 25 WT-PM models can be compared *within a rung* and not across
rungs by accident.

Every result in this repository states its rung.

---

## Rungs

| rung | data | labels | what a metric can claim |
| --- | --- | --- | --- |
| **0 — analytic** | closed-form or noiseless synthetic | exact | that the code implements the intended statistic |
| **1 — simulation** | physics-informed synthetic, full state | exact, injected at a known step | that the method works when its assumptions hold |
| **2 — replayed** | real SCADA, historical intervention record | from work orders / maintenance logs — incomplete, delayed, sometimes wrong | that the method survives real noise and real label noise |
| **3 — shadow** | real SCADA, live feed, no consequences | none; alarms reviewed by engineers | how often the method would have alerted, and how engineers judged it |
| **4 — closed loop** | real SCADA, alarms acted on | outcomes | avoided downtime and cost, against a counterfactual |

**This repository is at rung 1.** The generator (`simulate.py`) implements:
Heier's power coefficient with optimal TSR 8.1 and Cp 0.480, region-2
tip-speed-ratio tracking and region-3 pitch by bisection, thermal lags of 110
and 45 minutes, Rayleigh AR(1) wind with a nine-day synoptic component, explicit
mass and energy accounting, and Jensen wake deficits recomputed per timestep
from the met-mast wind direction. Physical envelopes are asserted in the
simulator's own tests (power ≤ rated, torque within the rated envelope, vibration
within its measured band).

Rung 1 buys things rung 2 cannot: exact onset times, exact fault families, and
the ability to run 60 days in 30 seconds. It does not buy realism, and no number
in this repository should be read as an estimate of field performance.

## What is reported, and in what order

1. **Raw point-wise precision / recall / F1** with a 95 % moving-block bootstrap
   interval. This is the headline because it is the number an operator's alert
   queue experiences.
2. **F1 confidence interval**, with normalised effective sample size. Blocks
   below one day cannot resolve day-scale weather autocorrelation; when the
   effective sample size falls under roughly ten blocks the interval is
   reported as unresolved rather than quoted as if it were informative.
3. **PR-AUC and ROC-AUC** — threshold-free, so they survive a change of
   operating point. PR-AUC is the more informative of the two at the base rates
   here (faults occupy ~30–60 % of the test band in the reference schedule,
   which is far above field prevalence; see the caveat below).
4. **Event recall and median latency in samples**, using a 6-hour detection
   tolerance. A detection that arrives after the maintenance window is not a
   detection, so latency is part of the result and not a footnote.
5. **Alarm time**: false-alarm minutes per day and total alarm minutes per day.
   Alarm *time* is the resource a crew spends; the false-alarm figure excludes
   the deliberate hysteresis tail after a real detection.
6. **Per-family recall**, because a detector that misses one failure mode
   entirely is not described by its average.
7. **An operating-point table** — the decision stage re-run at several
   multipliers of the calibrated threshold. It uses test-band labels and is
   labelled as evidence, never as a selection procedure.

## Base rates and why they matter

The reference run's test band is 64 % fault-labelled. Field prevalence is far
lower — a wind farm does not spend two-thirds of its life in a labelled fault
state. At low prevalence, precision collapses for a fixed recall: the same
detector at a 1 % fault rate turns a 0.75-precision operating point into an
alarm queue dominated by false positives.

`evaluate.py` therefore reports precision, recall and PR-AUC separately rather
than collapsing them into an accuracy-like number, and `false_alarms_per_day` is
a rate per *day* rather than per sample, so it can be compared against a crew's
capacity directly. Anyone quoting the F1 from this repository should also quote
the fault fraction — it is printed in every run report for that reason.

## Statistical honesty

* **No shuffled splits.** Train, calibration and test are contiguous bands. A
  random split of a time series leaks near-duplicate windows across the
  boundary and inflates every metric.
* **The scaler and the normal-behaviour model are fitted on train +
  calibration only**, the model on the train band only, and the threshold and
  component scales on the calibration band only. `split_indices` refuses to
  build a split whose train or calibration band contains a labelled fault.
* **Bootstrap before claiming a difference.** An F1 of 0.69 with an interval of
  [0.53, 0.81] does not distinguish itself from 0.60. Operating-point tables are
  for locating a trade-off, not for grading 0.02 differences.

## Reproducing a result

    wtpm demo --days 60 --epochs 30 --ensemble 2 --stride 2

writes `REPORT.md`, `score_trace.json`, a detector bundle and a run record under
`runs/<name>-<fingerprint>/`. The fingerprint is a hash of the full
configuration: two runs that share it are the same experiment, and the run
record carries the Python/Numpy/platform provenance alongside the metrics.
