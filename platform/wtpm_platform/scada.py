"""SCADA historian bridge — combine plant SCADA with the 25-model platform.

A SCADA system (or a scheduled historian export) can call:

    wt-pm scada --in historian.csv --turbine WT-07 --out tags.json

or POST the same payload to ``/scada``. Tag names from the plant are mapped
onto the 12-channel ``wt-pm.scada.v1`` contract; results are written back as
flat tags a SCADA point list can import (``WTPM.ALARM``, ``WTPM.RUL_H``, …).
"""

from __future__ import annotations

import csv
import json
import os
import re
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
    "yaw_error": "yaw_error_deg", "yaw": "yaw_error_deg", "nacelle_yaw_error": "yaw_error_deg",
    "power": "power_kw", "p_active": "power_kw", "active_power": "power_kw", "kw": "power_kw",
    "torque": "main_shaft_torque_knm", "shaft_torque": "main_shaft_torque_knm",
    "current": "generator_current_a", "stator_current": "generator_current_a",
    "igen": "generator_current_a",
    "frequency": "grid_frequency_hz", "grid_freq": "grid_frequency_hz", "f_grid": "grid_frequency_hz",
    "nacelle_temp": "nacelle_temp_c", "tnac": "nacelle_temp_c",
    "oil_temp": "gearbox_oil_temp_c", "gearbox_temp": "gearbox_oil_temp_c",
    "tgear": "gearbox_oil_temp_c", "gearbox_oil_temp": "gearbox_oil_temp_c",
    "vibration": "bearing_vib_rms_mm_s", "vib": "bearing_vib_rms_mm_s",
    "bearing_vib": "bearing_vib_rms_mm_s", "vib_rms": "bearing_vib_rms_mm_s",
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
    return {str(k): str(v) for k, v in raw.items()}


def probe_headers(headers: Sequence[str], extra: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Suggest a tag map from a historian CSV header row."""
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
    needed = [c for c in canonical_channels() if c not in mapped.values()]
    return {
        "mapped": mapped,
        "special": special,
        "unknown": unknown,
        "missing_canonical": needed,
        "coverage": round(len(set(mapped.values())) / max(len(canonical_channels()), 1), 3),
        "ready": len(mapped) >= 4,
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
    extra = dict(tag_map or {})
    headers = list(rows[0].keys())
    resolved: Dict[str, str] = {}
    time_col = None
    turb_col = None
    for h in headers:
        r = resolve_column(h, extra)
        if r == "__time__":
            time_col = h
        elif r == "__turbine__":
            turb_col = h
        elif r:
            resolved[h] = r
    if not resolved:
        raise ValueError(
            "no SCADA columns mapped onto the 12-channel contract; "
            "pass --map tag_map.json (plant_tag → canonical name)"
        )
    n = len(rows)
    chans = canonical_channels()
    values = np.full((n, len(chans)), np.nan)
    idx = {c: i for i, c in enumerate(chans)}
    for i, row in enumerate(rows):
        for h, canon in resolved.items():
            if canon not in idx:
                continue
            raw = row.get(h, "")
            if raw in (None, "", "NA", "NaN", "null"):
                continue
            try:
                values[i, idx[canon]] = float(raw)
            except (TypeError, ValueError):
                continue
        if turb_col and row.get(turb_col):
            turbine_id = str(row[turb_col])
    # physics fill: torque from P / ω if missing
    i_p, i_rpm, i_tq = idx["power_kw"], idx["rotor_speed_rpm"], idx["main_shaft_torque_knm"]
    miss_tq = ~np.isfinite(values[:, i_tq])
    if miss_tq.any() and np.isfinite(values[:, i_p]).any() and np.isfinite(values[:, i_rpm]).any():
        omega = np.maximum(values[:, i_rpm], 0.1) * 2 * np.pi / 60.0
        values[miss_tq, i_tq] = (values[miss_tq, i_p] / 0.94) / omega[miss_tq]  # kNm-ish
    ts = np.arange(n, dtype=np.int64) * 600
    if time_col:
        parsed = []
        for row in rows:
            parsed.append(_parse_time(row.get(time_col)))
        if all(t is not None for t in parsed):
            ts = np.asarray(parsed, dtype=np.int64)
    mapped = sorted(set(resolved.values()))
    batch = ingest_arrays(turbine_id, ts, values, chans)
    batch.meta["scada_mapped_channels"] = mapped
    batch.meta["scada_unmapped_headers"] = [h for h in headers if h not in resolved
                                            and h not in (time_col, turb_col)]
    batch.meta["source"] = "scada"
    batch.meta["scada_turbine_col"] = turb_col
    return batch


def split_by_turbine(
    rows: Sequence[Mapping[str, Any]],
    tag_map: Optional[Mapping[str, str]] = None,
    default_id: str = "WT-SCADA",
) -> Dict[str, SensorBatch]:
    """One SensorBatch per turbine_id column (fleet CSV from the historian)."""
    extra = dict(tag_map or {})
    turb_col = None
    if rows:
        for h in rows[0].keys():
            if resolve_column(h, extra) == "__turbine__":
                turb_col = h
                break
    if not turb_col:
        return {default_id: ingest_scada_rows(rows, turbine_id=default_id, tag_map=tag_map)}
    groups: Dict[str, list] = {}
    for row in rows:
        tid = str(row.get(turb_col) or default_id)
        groups.setdefault(tid, []).append(row)
    return {tid: ingest_scada_rows(g, turbine_id=tid, tag_map=tag_map)
            for tid, g in groups.items()}


def _parse_time(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        pass
    from datetime import datetime
    s = str(v).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%d/%m/%Y %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(s[:19].replace("Z", ""), fmt.replace("Z", "")).timestamp())
        except ValueError:
            continue
    return None


def emit_scada_tags(result: Mapping[str, Any], turbine_id: str) -> Dict[str, Any]:
    """Flat tags a SCADA point list / OPC UA client can write."""
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
