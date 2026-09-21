# wt-pm

Unified wind-turbine predictive-maintenance platform: **25 models, one orchestrator, one terminal command.**

```bash
pip install wt-pm
wt-pm inspect
wt-pm serve --port 8100
```

Until the package is on PyPI, install from this repo:

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
# or, from a clone:
pip install -e platform
```

Core extras (optional — missing deps just mark that adapter unavailable):

```bash
pip install "wt-pm[full]"     # torch, tensorflow, xgboost, shap, ...
```

## Terminal

```text
wt-pm --version
wt-pm inspect                          # 25-model registry + availability
wt-pm fetch-models                     # clone sibling research repos into ./external
wt-pm research --days 12               # train, held-out compare, fusion weights
wt-pm production --days 10 --out out.json
wt-pm fleet --turbines 6 --days 8
wt-pm edge --out edge_artifacts        # ESP32 model.h + INT8 tflite
wt-pm serve --port 8100                # API + command-center dashboard
wt-pm agent --days 8 --quiet           # Hermes Thought/Action/Observation + XAI
```

After `pip install`, the `wt-pm` executable is on your PATH (same as `wtpm-platform`).

Point the adapters at cloned research code:

```bash
wt-pm fetch-models --dir ./external
export WTPM_EXTERNAL_DIR=$PWD/external
```

A built-in SCADA simulator ships with the package so `research` / `production` / `serve` work without any extra repo. If `wt-pm-lstm-scada-anomaly` (model 05) is also installed, its physics simulator is used instead.
