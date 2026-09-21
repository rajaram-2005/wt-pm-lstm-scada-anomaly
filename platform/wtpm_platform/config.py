"""Plant config file so SCADA and the platform share one settings document.

Looks for ``wt-pm.yaml`` / ``wt-pm.json`` in cwd, ``$WTPM_CONFIG``, or ``--config``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "plant": "site-1",
    "drop_dir": "scada_in",
    "out_dir": "scada_out",
    "processed_dir": "scada_done",
    "map": "tag_map.json",
    "poll_seconds": 30,
    "serve_port": 8100,
    "workers": 4,
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    env = os.environ.get("WTPM_CONFIG")
    candidates = [path, env, "wt-pm.yaml", "wt-pm.yml", "wt-pm.json"]
    found = None
    for c in candidates:
        if c and os.path.isfile(c):
            found = c
            break
    if not found:
        return cfg
    with open(found, encoding="utf-8") as f:
        raw = f.read()
    if found.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(raw) or {}
        except ImportError:
            data = _simple_yaml(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"config {found} is not a mapping")
    cfg.update({k: v for k, v in data.items() if v is not None})
    cfg["_path"] = found
    return cfg


def _simple_yaml(text: str) -> Dict[str, Any]:
    """Minimal ``key: value`` parser so PyYAML is not required."""
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v) if "." not in v else float(v)
            except ValueError:
                out[k] = v
    return out
