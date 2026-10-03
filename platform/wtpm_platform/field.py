"""Open field-SCADA helpers (Hill of Towie / Zenodo-style column names)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

# Common HoT / UK wind-farm 10-min headers → canonical channels.
HOT_ALIASES: Dict[str, str] = {
    "WindSpeed": "wind_speed_ms",
    "ActivePower": "power_kw",
    "Power": "power_kw",
    "RotorSpeed": "rotor_speed_rpm",
    "PitchAngle": "pitch_angle_deg",
    "YawError": "yaw_error_deg",
    "MainShaftTorque": "main_shaft_torque_knm",
    "Torque": "main_shaft_torque_knm",
    "NacelleTemperature": "nacelle_temp_c",
    "GearOilTemperature": "gearbox_oil_temp_c",
    "AmbientTemperature": "ambient_temp_c",
    "GridFrequency": "grid_frequency_hz",
    "GeneratorCurrent": "generator_current_a",
    "BearingVibration": "bearing_vib_rms_mm_s",
    "VibrationRMS": "bearing_vib_rms_mm_s",
    "TimeStamp": "__time__",
    "TurbineName": "__turbine__",
    "Turbine": "__turbine__",
}


def write_hot_sample_csv(path: str, n: int = 48) -> str:
    """Tiny synthetic, 12-channel SCADA-shaped CSV (not field measurements)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = [
        "TimeStamp", "TurbineName", "WindSpeed", "AmbientTemperature",
        "RotorSpeed", "PitchAngle", "YawError", "ActivePower",
        "MainShaftTorque", "GeneratorCurrent", "GridFrequency",
        "NacelleTemperature", "GearOilTemperature", "BearingVibration",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(n):
            ws = 8.0 + 0.05 * i
            rpm = 10.0 + 0.4 * ws
            power = 120.0 * ws
            torque = power / (0.94 * rpm * 2.0 * 3.141592653589793 / 60.0)
            w.writerow([
                1_700_000_000 + i * 600, "T01", f"{ws:.3f}", "11.0",
                f"{rpm:.2f}", "2.0", f"{0.2 * ((i % 5) - 2):.2f}",
                f"{power:.1f}", f"{torque:.3f}", f"{power * 0.7:.1f}",
                "50.0", "28.0", "54.0", f"{2.0 + 0.01 * i:.3f}",
            ])
    return path


def hot_tag_map() -> Dict[str, str]:
    return dict(HOT_ALIASES)
