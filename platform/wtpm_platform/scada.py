"""SCADA historian ingestion and proposed result-tag formatting.

A historian export can be inspected with:

    wt-pm scada --in historian.csv --turbine WT-07 --out tags.json

or a trusted gateway can POST one turbine's rows to ``/scada``. Plant tags map
to the canonical 12-channel contract. The returned ``WTPM.*`` point list is a
candidate output for a separately reviewed integration; this module performs
no network write to a PLC, historian, OPC UA server or MQTT broker.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wtpm_platform.data import canonical_channels, ingest_arrays
from wtpm_platform.contracts import SensorBatch

# Plant tag aliases → canonical channel. Matching is case-insensitive after
# stripping units/underscores. Extend via --map JSON without code changes.
ALIASES: Dict[str, str] = {
    "wind_speed": "wind_speed_ms", "windspeed": "wind_speed_ms", "ws": "wind_speed_ms",
    "ws_ms": "wind_speed_ms", "hub_wind_speed": "wind_speed_ms", "anemometer": "wind_speed_ms",
    "ambient_temp": "ambient_temp_c", "tamb": "ambient_temp_c", "outside_temp": "ambient_temp_c",
    "rotor_speed": "rotor_speed_rpm", "rpm": "rotor_speed_rpm", "n_rotor": "rotor_speed_rpm",
    "pitch": "pitch_angle_deg", "pitch_angle": "pitch_angle_deg", "blade_pitch": "pitch_angle_deg",
    "yaw_error": "yaw_error_deg", "yawerror": "yaw_error_deg",
    "yaw": "yaw_error_deg", "nacelle_yaw_error": "yaw_error_deg",
    "power": "power_kw", "p_active": "power_kw", "active_power": "power_kw", "kw": "power_kw",
    "torque": "main_shaft_torque_knm", "shaft_torque": "main_shaft_torque_knm",
    "main_shaft_torque": "main_shaft_torque_knm",
    "mainshafttorque": "main_shaft_torque_knm",
    "current": "generator_current_a", "stator_current": "generator_current_a",
    "igen": "generator_current_a",
    "frequency": "grid_frequency_hz", "grid_freq": "grid_frequency_hz", "f_grid": "grid_frequency_hz",
    "nacelle_temp": "nacelle_temp_c", "tnac": "nacelle_temp_c",
    "oil_temp": "gearbox_oil_temp_c", "gearbox_temp": "gearbox_oil_temp_c",
    "tgear": "gearbox_oil_temp_c", "gearbox_oil_temp": "gearbox_oil_temp_c",
    "vibration": "bearing_vib_rms_mm_s", "vib": "bearing_vib_rms_mm_s",
    "bearing_vib": "bearing_vib_rms_mm_s", "bearingvibration": "bearing_vib_rms_mm_s",
    "vibration_rms": "bearing_vib_rms_mm_s", "vib_rms": "bearing_vib_rms_mm_s",
    "brgvib": "bearing_vib_rms_mm_s", "brg_vib": "bearing_vib_rms_mm_s",
    "activepower": "power_kw", "windspeed": "wind_speed_ms",
    "rotorspeed": "rotor_speed_rpm", "pitchangle": "pitch_angle_deg",
    "gearoiltemperature": "gearbox_oil_temp_c",
    "nacelletemperature": "nacelle_temp_c",
    "ambienttemperature": "ambient_temp_c",
    "gridfrequency": "grid_frequency_hz",
    "generatorcurrent": "generator_current_a",
    "turbinename": "__turbine__",
    "gboiltemp": "gearbox_oil_temp_c", "gboil": "gearbox_oil_temp_c",
    "timestamp": "__time__", "time": "__time__", "ts": "__time__", "epoch": "__time__",
    "datetime": "__time__", "date": "__time__",
    "turbine": "__turbine__", "turbine_id": "__turbine__", "wt": "__turbine__", "asset": "__turbine__",
}

WRITEBACK_TAGS = (
    "WTPM.ALARM", "WTPM.SCORE", "WTPM.RISK", "WTPM.HEALTH",
    "WTPM.FAULT", "WTPM.SUBSYSTEM", "WTPM.RUL_H", "WTPM.ACTION",
    "WTPM.SAFETY", "WTPM.TRUST",
)

PG_BNN_REQUIRED_CHANNELS = (
    "wind_speed_ms", "ambient_temp_c", "rotor_speed_rpm",
    "pitch_angle_deg", "main_shaft_torque_knm", "power_kw",
)


def _validated_tag_map(tag_map: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Validate a plant-tag → canonical-channel map before using it."""
    if tag_map is None:
        return {}
    if not isinstance(tag_map, Mapping):
        raise ValueError("tag map must be an object mapping plant tags to canonical channels")
    allowed = set(canonical_channels()) | {"__time__", "__turbine__"}
    out: Dict[str, str] = {}
    normalized = set()
    for raw_key, raw_value in tag_map.items():
        key, value = str(raw_key).strip(), str(raw_value).strip()
        if not key:
            raise ValueError("tag map contains an empty plant tag")
        if value not in allowed:
            raise ValueError(
                f"tag map target {value!r} for {key!r} is not a canonical channel, "
                "__time__, or __turbine__"
            )
        norm_key = _norm(key)
        if norm_key in normalized:
            raise ValueError(f"tag map contains duplicate normalized plant tag: {key!r}")
        normalized.add(norm_key)
        out[key] = value
    return out


def _norm(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\[.*?\]", "", s)
    s = s.replace("°", "").replace("%", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    for suf in ("_ms", "_c", "_deg", "_kw", "_rpm", "_hz", "_a", "_knm", "_mm_s"):
        if s.endswith(suf) and s[: -len(suf)]:
            pass
    return s


def resolve_column(col: str, extra: Optional[Mapping[str, str]] = None) -> Optional[str]:
    n = _norm(col)
    n_stripped = re.sub(r"^(wt|turbine|unit)\d*_", "", n)
    if extra:
        extra = _validated_tag_map(extra)
        extra_n = {_norm(k): v for k, v in extra.items()}
        if n in extra_n:
            return extra_n[n]
        if n_stripped in extra_n:
            return extra_n[n_stripped]
        if col in extra:
            return extra[col]
    for cand in (n, n_stripped):
        if cand in ALIASES:
            return ALIASES[cand]
    # already canonical
    if n in canonical_channels() or col in canonical_channels():
        return n if n in canonical_channels() else col
    # fuzzy: channel name contained
    for ch in canonical_channels():
        if n == _norm(ch) or n.replace("_", "") == ch.replace("_", ""):
            return ch
    return None


def load_tag_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _validated_tag_map(raw)


def probe_headers(headers: Sequence[str], extra: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Suggest a tag map and report readiness for m18 and all 12 channels."""
    extra = _validated_tag_map(extra)
    mapped, unknown, special = {}, [], {}
    for h in headers:
        r = resolve_column(h, extra)
        if r == "__time__":
            special["time"] = h
        elif r == "__turbine__":
            special["turbine"] = h
        elif r:
            mapped[h] = r
        else:
            unknown.append(h)
    present = set(mapped.values())
    needed = [c for c in canonical_channels() if c not in present]
    missing_pg_bnn = [c for c in PG_BNN_REQUIRED_CHANNELS if c not in present]
    counts = {channel: list(mapped.values()).count(channel) for channel in present}
    ambiguous = {channel: count for channel, count in counts.items() if count > 1}
    has_time = "time" in special
    full_ready = not needed and not ambiguous and has_time
    pgbnn_ready = has_time and not missing_pg_bnn and not any(
        channel in ambiguous for channel in PG_BNN_REQUIRED_CHANNELS
    )
    return {
        "mapped": mapped,
        "special": special,
        "unknown": unknown,
        "missing_canonical": needed,
        "missing_pg_bnn": missing_pg_bnn,
        "ambiguous_channels": ambiguous,
        "coverage": round(len(present) / max(len(canonical_channels()), 1), 3),
        # Kept for compatibility: ready means the input can be parsed, not that
        # it has enough sensors for every model or is approved for field use.
        "ready": len(mapped) >= 4,
        "ready_for_pg_bnn": pgbnn_ready,
        "ready_for_full_profile": full_ready,
        "time_mapped": "time" in special,
        "turbine_mapped": "turbine" in special,
    }


def ingest_scada_csv(
    path: str,
    turbine_id: str = "WT-SCADA",
    tag_map: Optional[Mapping[str, str]] = None,
) -> SensorBatch:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty SCADA CSV: {path}")
    return ingest_scada_rows(rows, turbine_id=turbine_id, tag_map=tag_map)


def ingest_scada_rows(
    rows: Sequence[Mapping[str, Any]],
    turbine_id: str = "WT-SCADA",
    tag_map: Optional[Mapping[str, str]] = None,
) -> SensorBatch:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("SCADA rows must be a non-empty list of row objects")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("every SCADA row must be an object mapping tags to values")
    headers = list(dict.fromkeys(key for row in rows for key in row.keys()))
    if not all(isinstance(h, str) for h in headers):
        raise ValueError("SCADA tag names must be strings")

    extra = _validated_tag_map(tag_map)
    resolved: Dict[str, str] = {}
    time_cols: List[str] = []
    turbine_cols: List[str] = []
    for h in headers:
        r = resolve_column(h, extra)
        if r == "__time__":
            time_cols.append(h)
        elif r == "__turbine__":
            turbine_cols.append(h)
        elif r:
            resolved[h] = r
    if len(time_cols) > 1:
        raise ValueError(f"multiple SCADA columns map to time: {time_cols}")
    if len(turbine_cols) > 1:
        raise ValueError(f"multiple SCADA columns map to turbine id: {turbine_cols}")
    if not resolved:
        raise ValueError(
            "no SCADA columns mapped onto the 12-channel contract; "
            "pass a plant-tag → canonical-channel map"
        )

    sources_by_channel: Dict[str, List[str]] = {}
    for source, channel in resolved.items():
        sources_by_channel.setdefault(channel, []).append(source)
    ambiguous = {channel: sources for channel, sources in sources_by_channel.items()
                 if len(sources) > 1}
    if ambiguous:
        raise ValueError(f"multiple SCADA tags map to the same channel: {ambiguous}")

    time_col = time_cols[0] if time_cols else None
    turb_col = turbine_cols[0] if turbine_cols else None
    if turb_col:
        turbine_ids = {str(row.get(turb_col)).strip() for row in rows
                       if row.get(turb_col) not in (None, "")}
        if len(turbine_ids) > 1:
            raise ValueError(
                "a SCADA request must contain one turbine; split fleet rows by turbine_id first"
            )
        if turbine_ids:
            turbine_id = next(iter(turbine_ids))

    n = len(rows)
    chans = canonical_channels()
    values = np.full((n, len(chans)), np.nan)
    idx = {c: i for i, c in enumerate(chans)}
    for i, row in enumerate(rows):
        for h, canon in resolved.items():
            raw = row.get(h, "")
            if raw is None or (isinstance(raw, str) and raw.strip().lower() in
                               ("", "na", "nan", "null")):
                continue
            try:
                measurement = float(raw)
            except (TypeError, ValueError):
                # Invalid measurements remain missing and are quality-masked;
                # a production gateway must alarm on this upstream.
                continue
            if np.isfinite(measurement):
                values[i, idx[canon]] = measurement
    if not np.isfinite(values).any():
        raise ValueError("SCADA rows contain no finite numeric measurements")

    # Torque can be estimated from measured power and rotor speed when absent.
    # This is a convenience feature, not a measured torque replacement.
    i_p, i_rpm, i_tq = idx["power_kw"], idx["rotor_speed_rpm"], idx["main_shaft_torque_knm"]
    miss_tq = ~np.isfinite(values[:, i_tq])
    valid_power_rpm = np.isfinite(values[:, i_p]) & np.isfinite(values[:, i_rpm])
    fill_tq = miss_tq & valid_power_rpm & (values[:, i_rpm] > 0)
    if fill_tq.any():
        omega = values[:, i_rpm] * 2 * np.pi / 60.0
        values[fill_tq, i_tq] = (values[fill_tq, i_p] / 0.94) / omega[fill_tq]

    timestamps_generated = time_col is None
    if time_col:
        parsed = [_parse_time(row.get(time_col)) for row in rows]
        if any(t is None for t in parsed):
            raise ValueError(
                f"timestamp column {time_col!r} contains invalid or timezone-naive values; "
                "use UTC epoch seconds/milliseconds or timezone-aware ISO-8601"
            )
        ts = np.asarray(parsed, dtype=np.int64)
    else:
        # Backwards-compatible synthetic cadence for offline previews only.
        ts = np.arange(n, dtype=np.int64) * 600
    if n > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError(
            "SCADA timestamps must be strictly increasing and unique per turbine; "
            "sort and deduplicate rows before ingestion"
        )

    mapped = sorted(set(resolved.values()))
    batch = ingest_arrays(turbine_id, ts, values, chans)
    batch.meta["scada_mapped_channels"] = mapped
    batch.meta["scada_unmapped_headers"] = [h for h in headers if h not in resolved
                                            and h not in (time_col, turb_col)]
    batch.meta["scada_timestamps_generated"] = timestamps_generated
    batch.meta["source"] = "scada"
    batch.meta["scada_turbine_col"] = turb_col
    return batch


def split_by_turbine(
    rows: Sequence[Mapping[str, Any]],
    tag_map: Optional[Mapping[str, str]] = None,
    default_id: str = "WT-SCADA",
) -> Dict[str, SensorBatch]:
    """One SensorBatch per turbine_id column (fleet CSV from the historian)."""
    if not rows:
        raise ValueError("SCADA rows must be a non-empty list")
    extra = _validated_tag_map(tag_map)
    headers = list(dict.fromkeys(key for row in rows for key in row.keys()))
    turb_cols = [h for h in headers if resolve_column(h, extra) == "__turbine__"]
    if len(turb_cols) > 1:
        raise ValueError(f"multiple SCADA columns map to turbine id: {turb_cols}")
    if not turb_cols:
        return {default_id: ingest_scada_rows(rows, turbine_id=default_id, tag_map=extra)}
    turb_col = turb_cols[0]
    groups: Dict[str, list] = {}
    for row in rows:
        tid = str(row.get(turb_col) or default_id).strip()
        groups.setdefault(tid, []).append(row)
    return {tid: ingest_scada_rows(g, turbine_id=tid, tag_map=extra)
            for tid, g in groups.items()}


def _parse_epoch(value: Real) -> Optional[int]:
    number = float(value)
    if not np.isfinite(number):
        return None
    magnitude = abs(number)
    if magnitude >= 1e17:       # nanoseconds
        number /= 1e9
    elif magnitude >= 1e14:     # microseconds
        number /= 1e6
    elif magnitude >= 1e11:     # milliseconds
        number /= 1e3
    return int(round(number))


def _parse_time(v: Any) -> Optional[int]:
    """Parse Unix seconds/subseconds or timezone-aware ISO-8601 into UTC seconds."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, Real):
        return _parse_epoch(v)
    s = str(v).strip()
    try:
        return _parse_epoch(float(s))
    except ValueError:
        pass
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return int(parsed.timestamp())


def emit_scada_tags(result: Mapping[str, Any], turbine_id: str) -> Dict[str, Any]:
    """Format advisory WTPM tags; does not perform any SCADA network write."""
    what = result.get("what") or {}
    where = result.get("where") or {}
    rul = result.get("rul") or {}
    action = result.get("action") or {}
    safety = result.get("safety") or {}
    hidx = result.get("health_index") or {}
    quality = result.get("sensor_quality") or {}
    tags = {
        "WTPM.TURBINE": turbine_id,
        "WTPM.ALARM": 1 if what.get("alarm") else 0,
        "WTPM.SCORE": what.get("current_fused_score"),
        "WTPM.RISK": result.get("risk_score"),
        "WTPM.HEALTH": (hidx or {}).get("now"),
        "WTPM.FAULT": what.get("fault"),
        "WTPM.FAULT_CONF": what.get("fault_confidence"),
        "WTPM.SUBSYSTEM": where.get("subsystem"),
        "WTPM.RUL_H": rul.get("hours"),
        "WTPM.RUL_UNC_H": rul.get("uncertainty_hours"),
        "WTPM.ACTION": action.get("action"),
        "WTPM.SAFETY": safety.get("decision"),
        "WTPM.TRUST": quality.get("trust"),
        "WTPM.NARRATIVE": ((result.get("why") or {}).get("narrative")
                           or ((result.get("hermes") or {}).get("final") or {}).get("why", {}).get("narrative")),
    }
    return {
        "schema": "wt-pm.scada.writeback.v1",
        "turbine_id": turbine_id,
        "timestamp": result.get("timestamp"),
        "tags": tags,
        "point_list": [
            {"tag": k, "value": v, "quality": "good" if v is not None else "bad"}
            for k, v in tags.items()
        ],
    }


def write_tags_csv(payload: Mapping[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "value", "quality", "turbine_id", "timestamp"])
        for p in payload["point_list"]:
            w.writerow([p["tag"], p["value"], p["quality"],
                        payload["turbine_id"], payload.get("timestamp")])
    return path


def default_map_document() -> Dict[str, str]:
    """Example plant-tag → canonical map operators drop next to the historian."""
    return {
        "WT07.WindSpeed": "wind_speed_ms",
        "WT07.AmbTemp": "ambient_temp_c",
        "WT07.RotorRPM": "rotor_speed_rpm",
        "WT07.Pitch": "pitch_angle_deg",
        "WT07.YawErr": "yaw_error_deg",
        "WT07.Pwr_kW": "power_kw",
        "WT07.Torque": "main_shaft_torque_knm",
        "WT07.GenI": "generator_current_a",
        "WT07.GridHz": "grid_frequency_hz",
        "WT07.NacTemp": "nacelle_temp_c",
        "WT07.GbOilTemp": "gearbox_oil_temp_c",
        "WT07.BrgVib": "bearing_vib_rms_mm_s",
        "DateTime": "__time__",
        "Turbine": "__turbine__",
    }
