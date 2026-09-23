"""Software SIL proxy for m25 hard limits.

AI / SCADA writebacks pass through ``gate_command``. Anything that would
overspeed the rotor or command a negative pitch is dropped — same idea as
the ESP32 safety relay, without hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import math


# Demonstration engineering bounds (software emulator, not a certified PLC).
MAX_ROTOR_RPM = 25.0
MIN_PITCH_DEG = 0.0
MAX_PITCH_DEG = 90.0
MAX_VIB_MM_S = 20.0


def safety_gate_check(pitch_angle: Optional[float] = None,
                      rpm: Optional[float] = None,
                      vibration_mm_s: Optional[float] = None) -> Dict[str, Any]:
    reasons = []
    values = {}
    for name, value in (("pitch_angle", pitch_angle), ("rpm", rpm),
                        ("vibration_mm_s", vibration_mm_s)):
        if value is None:
            values[name] = None
            continue
        try:
            if isinstance(value, bool):
                raise ValueError("boolean is not a sensor value")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("non-finite value")
            values[name] = value
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"invalid {name}")
            values[name] = None
    pitch_angle, rpm, vibration_mm_s = (values[k] for k in
                                      ("pitch_angle", "rpm", "vibration_mm_s"))
    if rpm is not None and rpm < 0:
        reasons.append("negative rpm")
    if vibration_mm_s is not None and vibration_mm_s < 0:
        reasons.append("negative vibration")
    if rpm is not None and rpm > MAX_ROTOR_RPM:
        reasons.append(f"rpm {rpm} > {MAX_ROTOR_RPM} (overspeed)")
    if pitch_angle is not None and pitch_angle < MIN_PITCH_DEG:
        reasons.append(f"pitch {pitch_angle} < {MIN_PITCH_DEG}")
    if pitch_angle is not None and pitch_angle > MAX_PITCH_DEG:
        reasons.append(f"pitch {pitch_angle} > {MAX_PITCH_DEG}")
    if vibration_mm_s is not None and vibration_mm_s > MAX_VIB_MM_S:
        reasons.append(f"vib {vibration_mm_s} > {MAX_VIB_MM_S} mm/s")
    ok = not reasons
    return {
        "ok": ok,
        "dropped": not ok,
        "reasons": reasons,
        "message": ("CRITICAL: Command dropped. Hard limit exceeded."
                    if reasons else "Software check passed; no SCADA write performed"),
    }


def gate_command(cmd: Mapping[str, Any]) -> Dict[str, Any]:
    """Passthrough dict if limits hold; else drop with reasons."""
    # Check every supplied alias: zero is a value, and conflicting aliases may
    # never conceal an unsafe input. This function does not control hardware.
    aliases = {"pitch_angle": "pitch_angle", "pitch_angle_deg": "pitch_angle",
               "rpm": "rpm", "rotor_speed_rpm": "rpm",
               "vibration": "vibration_mm_s", "bearing_vib_rms_mm_s": "vibration_mm_s"}
    reasons = []
    supplied = [(key, arg) for key, arg in aliases.items() if key in cmd]
    if not supplied:
        reasons.append("no recognized safety parameters")
    for key, arg in supplied:
        if cmd[key] is None:
            reasons.append(f"invalid {key}")
        else:
            reasons.extend(safety_gate_check(**{arg: cmd[key]})["reasons"])
    check = {"ok": not reasons, "dropped": bool(reasons), "reasons": reasons,
             "message": ("CRITICAL: Command dropped. Invalid input or hard limit exceeded."
                         if reasons else "Software check passed; no SCADA write performed")}
    return {**cmd, "sil": check, "forwarded": not reasons}
