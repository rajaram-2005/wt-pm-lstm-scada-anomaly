"""Local protocol emulators (no plant, no token).

Stdlib JSON/TCP stand-ins so ingest can be wired before pymodbus / asyncua /
paho-mqtt are on the box. Optional real libraries are used when importable.

Register map (holding / OPC nodes / MQTT JSON keys):
  0 wind_speed_ms, 1 ambient_temp_c, 2 rotor_speed_rpm, 3 pitch_angle_deg,
  4 yaw_error_deg, 5 power_kw, 6 main_shaft_torque_knm, 7 generator_current_a,
  8 grid_frequency_hz, 9 nacelle_temp_c, 10 gearbox_oil_temp_c,
  11 bearing_vib_rms_mm_s
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Dict, List, Optional, Sequence

from wtpm_platform.data import canonical_channels, ingest_arrays
from wtpm_platform.contracts import SensorBatch

DEFAULT_HOLDING = [80, 120, 120, 20, 5, 1500, 30, 80, 500, 300, 75, 12]
# scaled ×10 except as documented; snapshot() divides by 10 for engineering units.


def engineering_from_holding(regs: Sequence[int]) -> Dict[str, float]:
    chans = canonical_channels()
    vals = [float(r) / 10.0 for r in list(regs)[:12]]
    while len(vals) < 12:
        vals.append(float("nan"))
    return {chans[i]: vals[i] for i in range(12)}


def snapshot_to_batch(regs: Sequence[int], turbine_id: str = "WT-MOCK",
                      ts: int = 1_700_000_000) -> SensorBatch:
    eng = engineering_from_holding(regs)
    chans = canonical_channels()
    import numpy as np
    values = np.array([[eng[c] for c in chans]], dtype=float)
    return ingest_arrays(turbine_id, np.array([ts], dtype=np.int64), values, chans)


class InMemoryBus:
    """Shared register file used by the stdlib TCP mock and unit tests."""

    def __init__(self, holding: Optional[List[int]] = None):
        self.holding = list(holding or DEFAULT_HOLDING)

    def read_hr(self, addr: int = 0, count: int = 12) -> List[int]:
        return self.holding[addr: addr + count]

    def write_hr(self, addr: int, values: Sequence[int]) -> None:
        for i, v in enumerate(values):
            while addr + i >= len(self.holding):
                self.holding.append(0)
            self.holding[addr + i] = int(v)


class JsonTcpServer:
    """Tiny JSON-lines TCP server: {\"op\":\"read\"} / {\"op\":\"write\",\"addr\":0,\"values\":[...]}."""

    def __init__(self, bus: Optional[InMemoryBus] = None, host: str = "0.0.0.0", port: int = 15020):
        self.bus = bus or InMemoryBus()
        self.host, self.port = host, port
        self._sock: Optional[socket.socket] = None
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> "JsonTcpServer":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)

    def _loop(self) -> None:
        assert self._sock
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                buf = b""
                conn.settimeout(2)
                try:
                    while b"\n" not in buf:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    if not buf:
                        continue
                    req = json.loads(buf.decode("utf-8").split("\n", 1)[0])
                    conn.sendall((json.dumps(self._handle(req)) + "\n").encode())
                except Exception as exc:  # noqa: BLE001
                    try:
                        conn.sendall((json.dumps({"error": str(exc)}) + "\n").encode())
                    except OSError:
                        pass

    def _handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "read":
            addr = int(req.get("addr", 0))
            n = int(req.get("count", 12))
            regs = self.bus.read_hr(addr, n)
            return {"regs": regs, "engineering": engineering_from_holding(regs)}
        if op == "write":
            from wtpm_platform.safety_gate import gate_command
            vals = req.get("values") or []
            gated = gate_command({
                "rpm": (vals[2] / 10.0) if len(vals) > 2 else None,
                "pitch_angle": (vals[3] / 10.0) if len(vals) > 3 else None,
            })
            if not gated["forwarded"]:
                return {"ok": False, "sil": gated["sil"]}
            self.bus.write_hr(int(req.get("addr", 0)), vals)
            return {"ok": True}
        return {"error": "unknown op"}


def client_read(host: str = "127.0.0.1", port: int = 15020) -> dict:
    s = socket.create_connection((host, port), timeout=3)
    with s:
        s.sendall(b'{"op":"read"}\n')
        return json.loads(s.recv(8192).decode())


def try_pymodbus_server(port: int = 5020):
    """Start pymodbus if installed; else return None."""
    try:
        from pymodbus.datastore import (
            ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext,
        )
        from pymodbus.server import StartTcpServer
    except ImportError:
        return None
    store = ModbusSlaveContext(hr=ModbusSequentialDataBlock(0, list(DEFAULT_HOLDING)))
    ctx = ModbusServerContext(slaves=store, single=True)
    return StartTcpServer, ctx, ("0.0.0.0", port)
