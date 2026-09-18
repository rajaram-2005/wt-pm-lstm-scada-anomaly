"""WT-PM model 13 — LSTM sequence anomaly detection on wind-turbine SCADA.

Position in the WT-PM intelligence fabric (see ``docs/architecture.md``):

    Layer 0   physical asset                -> :mod:`.simulate` (the simulated asset)
    Layer 1   multi-modal data acquisition  -> :mod:`.schema`, :mod:`.dataio`
    Layer 2   data engineering              -> :mod:`.windows`
    Layer 4   anomaly intelligence          -> :mod:`.baseline`, :mod:`.nn`, :mod:`.detect`
    Layer 6   temporal degradation engine   -> :mod:`.detect` (trend component), :mod:`.drift`
    Layer 10  fusion and decision           -> :mod:`.detect` (four-component fusion)
    Layer 12  explainability                -> :mod:`.detect` (per-channel attribution)
    Layer 13  operations                    -> :mod:`.report`, :mod:`.api`, :mod:`.cli`

Every cross-repository boundary is an explicit, versioned contract
(:data:`~wt_pm_lstm.schema.SCHEMA_VERSION` going in,
:data:`~wt_pm_lstm.schema.ANOMALY_SCHEMA_VERSION` coming out), so the other
WT-PM models can consume this one without importing its internals.

Quick start::

    wtpm demo --days 60 --epochs 30      # full protocol, writes a report
    wtpm serve                           # contract-shaped HTTP API
    wtpm score --detector bundle --csv scada.csv
"""

from wt_pm_lstm.schema import (
    ANOMALY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnomalyRecord,
    ChannelSpec,
    TurbineTimeline,
    canonical_channels,
    default_score_channels,
    get_channel,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "ANOMALY_SCHEMA_VERSION",
    "AnomalyRecord",
    "ChannelSpec",
    "TurbineTimeline",
    "canonical_channels",
    "default_score_channels",
    "get_channel",
]
