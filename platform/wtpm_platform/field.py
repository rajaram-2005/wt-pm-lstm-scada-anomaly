"""Open field-SCADA helpers (Hill of Towie / Zenodo-style column names)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

# Common HoT / UK wind-farm 10-min headers → canonical channels.
HOT_ALIASES: Dict[str, str] = {
    "WindSpeed": "wind_speed_ms",
    "windspeed": "wind_speed_ms",
    "ActivePower": "power_kw",
    "activepower": "power_kw",
    "Power": "power_kw",
    "RotorSpeed": "rotor_speed_rpm",
    "rotorspeed": "rotor_speed_rpm",
    "PitchAngle": "pitch_angle_deg",
    "NacelleTemperature": "nacelle_temp_c",
    "GearOilTemperature": "gearbox_oil_temp_c",
    "AmbientTemperature": "ambient_temp_c",
    "GridFrequency": "grid_frequency_hz",
    "GeneratorCurrent": "generator_current_a",
    "TimeStamp": "__time__",
    "timestamp": "__time__",
    "TurbineName": "__turbine__",
    "Turbine": "__turbine__",
}


def write_hot_sample_csv(path: str, n: int = 48) -> str:
    """Tiny synthetic Hill-of-Towie-shaped CSV (no Zenodo download in CI)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = ["TimeStamp", "TurbineName", "WindSpeed", "ActivePower",
               "RotorSpeed", "PitchAngle", "AmbientTemperature",
               "GearOilTemperature", "NacelleTemperature"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(n):
            ws = 8.0 + 0.05 * i
            w.writerow([
                1_700_000_000 + i * 600, "T01", f"{ws:.3f}", f"{120*ws:.1f}",
                f"{10+0.4*ws:.2f}", "2.0", "11.0", "54.0", "28.0",
            ])
    return path


def hot_tag_map() -> Dict[str, str]:
    return dict(HOT_ALIASES)
