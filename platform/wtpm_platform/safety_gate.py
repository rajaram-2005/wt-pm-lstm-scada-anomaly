"""Software SIL proxy for m25 hard limits.

AI / SCADA writebacks pass through ``gate_command``. Anything that would
overspeed the rotor or command a negative pitch is dropped — same idea as
the ESP32 safety relay, without hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


# User-specified SIL-3-style bounds (software emulator, not a certified PLC).
MAX_ROTOR_RPM = 25.0
MIN_PITCH_DEG = 0.0
MAX_PITCH_DEG = 90.0
MAX_VIB_MM_S = 20.0


def safety_gate_check(pitch_angle: Optional[float] = None,
                      rpm: Optional[float] = None,
                      vibration_mm_s: Optional[float] = None) -> Dict[str, Any]:
    reasons = []
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
                    if reasons else "Command passed to SCADA"),
    }


def gate_command(cmd: Mapping[str, Any]) -> Dict[str, Any]:
    """Passthrough dict if limits hold; else drop with reasons."""
    check = safety_gate_check(
        pitch_angle=_num(cmd.get("pitch_angle") or cmd.get("pitch_angle_deg")),
        rpm=_num(cmd.get("rpm") or cmd.get("rotor_speed_rpm")),
        vibration_mm_s=_num(cmd.get("vibration") or cmd.get("bearing_vib_rms_mm_s")),
    )
    out = dict(cmd)
    out["sil"] = check
    if check["dropped"]:
        out["forwarded"] = False
    else:
        out["forwarded"] = True
    return out


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
