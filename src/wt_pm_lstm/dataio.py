"""Reading and writing the data contract (CSV/JSON in, artefacts out).

A detector that can only consume its own simulator's output is a demo, not a
tool. :func:`read_csv_timeline` loads the flat CSV layout written by
:meth:`~wt_pm_lstm.schema.TurbineTimeline.to_csv` — timestamp, one column per
channel, then ``q_<channel>`` quality columns — and rebuilds a conformant
timeline, flagging anything it cannot trust.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wt_pm_lstm.schema import (
    QUALITY_MISSING,
    QUALITY_OK,
    QUALITY_OUT_OF_RANGE,
    TurbineTimeline,
    canonical_channels,
    get_channel,
)

Array = np.ndarray


def read_csv_timeline(
    path: str,
    turbine_id: str = "WT-001",
    site_id: str = "unknown-site",
    channel_names: Optional[Sequence[str]] = None,
    timestamp_unit: str = "s",
) -> TurbineTimeline:
    """Load a SCADA CSV into a :class:`TurbineTimeline`.

    Rules applied on load, so that downstream code never has to guess:

    * an empty channel cell becomes ``mask=False`` and ``QUALITY_MISSING``;
    * a value outside the channel's physical envelope is kept but flagged
      ``QUALITY_OUT_OF_RANGE`` (deleting it would hide a real sensor fault);
    * timestamps are taken as integer seconds, milliseconds or ISO-8601 and
      converted to Unix seconds.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    if header[0].lower() not in ("timestamp", "time", "ts"):
        raise ValueError(f"{path}: first column must be a timestamp, got {header[0]!r}")

    names = tuple(channel_names) if channel_names else tuple(
        h for h in header[1:] if not h.startswith("q_")
    )
    quality_cols = {h.replace("q_", ""): i for i, h in enumerate(header) if h.startswith("q_")}

    n, d = len(rows), len(names)
    values = np.full((n, d), np.nan)
    mask = np.ones((n, d), dtype=bool)
    quality = np.full((n, d), QUALITY_OK, dtype=np.uint8)
    timestamps = np.zeros(n, dtype=np.int64)

    for i, row in enumerate(rows):
        timestamps[i] = _parse_timestamp(row[0], timestamp_unit)
        for j, name in enumerate(names):
            idx = header.index(name)
            cell = row[idx].strip() if idx < len(row) else ""
            if cell == "":
                values[i, j] = 0.0
                mask[i, j] = False
                quality[i, j] = QUALITY_MISSING
                continue
            v = float(cell)
            values[i, j] = v
            if not np.isfinite(v):
                mask[i, j] = False
                quality[i, j] = QUALITY_MISSING
                continue
            spec = get_channel(name)
            if not spec.in_range(v):
                quality[i, j] = QUALITY_OUT_OF_RANGE
            if name in quality_cols:
                declared = int(float(row[quality_cols[name]]))
                quality[i, j] = declared
                if declared in (1, 5):
                    mask[i, j] = False
    return TurbineTimeline(
        turbine_id=turbine_id,
        site_id=site_id,
        channel_names=names,
        values=values,
        timestamps=timestamps,
        mask=mask,
        quality=quality,
        meta={"source": os.path.basename(path), "schema": "wt-pm.scada.v1"},
    )


def _parse_timestamp(cell: str, unit: str) -> int:
    cell = cell.strip()
    if cell.isdigit() or (cell.startswith("-") and cell[1:].isdigit()):
        value = int(cell)
        if unit in ("ms", "milliseconds"):
            return value // 1000
        if unit in ("us", "microseconds"):
            return value // 1_000_000
        return value
    # ISO-8601 without pulling in a datetime dependency chain at import time.
    from datetime import datetime, timezone

    text = cell.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def write_json(path: str, payload: Any) -> str:
    """Write JSON, creating parent directories, in a stable key order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_default)
        fh.write("\n")
    return path


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def records_to_json(records: Sequence[Any]) -> List[Dict[str, Any]]:
    return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]


__all__ = ["read_csv_timeline", "write_json", "records_to_json"]
