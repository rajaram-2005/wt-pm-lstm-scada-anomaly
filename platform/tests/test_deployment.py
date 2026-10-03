"""Cloud deployment contract: PORT and a real readiness probe."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from wtpm_platform import api, cli


@pytest.mark.parametrize("port,extra,expected", [
    (None, [], 8100),
    ("10000", [], 10000),
    ("10000", ["--port", "8200"], 8200),
])
def test_serve_port(monkeypatch, port, extra, expected):
    if port is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", port)
    calls = []
    monkeypatch.setattr(api, "serve", lambda **kwargs: calls.append(kwargs))
    assert cli.main(["serve", *extra]) == 0
    assert calls[0]["port"] == expected


def test_readiness_during_bootstrap(monkeypatch):
    monkeypatch.setitem(api._STATE, "fitted", False)
    monkeypatch.setitem(api._STATE, "orchestrator", None)
    server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/ready", timeout=5)
        assert error.value.code == 503
        assert json.loads(error.value.read()) == {"status": "starting"}
        error.value.close()
        with urlopen(base + "/health", timeout=5) as response:
            assert response.status == 200
        monkeypatch.setitem(api._STATE, "fitted", True)
        with urlopen(base + "/ready", timeout=5) as response:
            assert response.status == 200
            assert json.load(response) == {"status": "ready"}
        with urlopen(base + "/", timeout=5) as response:
            assert response.status == 200
            assert b"<!DOCTYPE html>" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cors_requires_an_exact_configured_origin(monkeypatch):
    monkeypatch.setenv("WTPM_CORS_ORIGINS", "https://ops.example,https://scada.example")
    monkeypatch.setattr(api, "_STATE", {
        "orchestrator": None, "fitted": False, "error": None,
    })
    server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        allowed = Request(base + "/health", headers={"Origin": "https://ops.example"})
        with urlopen(allowed, timeout=5) as response:
            assert response.headers["Access-Control-Allow-Origin"] == "https://ops.example"

        denied = Request(base + "/health", headers={"Origin": "https://untrusted.example"})
        with urlopen(denied, timeout=5) as response:
            assert response.headers.get("Access-Control-Allow-Origin") is None

        preflight = Request(
            base + "/scada", method="OPTIONS",
            headers={"Origin": "https://untrusted.example"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(preflight, timeout=5)
        assert error.value.code == 403
        error.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_scada_api_requires_source_timestamps(monkeypatch):
    monkeypatch.setattr(api, "_STATE", {
        "orchestrator": object(), "batch": None, "ctx": None,
        "fitted": True, "lock": threading.Lock(), "last": None,
        "feedback": [], "error": None,
    })
    server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    payload = json.dumps({"turbine_id": "WT-07", "rows": [
        {"wind_speed": 8.0, "power": 100.0, "rpm": 10.0},
    ]}).encode()
    request = Request(
        base + "/scada", data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
        body = json.loads(error.value.read())
        assert "source timestamp" in body["error"]
        error.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_scada_api_returns_candidate_tags_for_one_turbine_only(monkeypatch):
    class FakeAudit:
        def record(self, *_args, **_kwargs):
            pass

    class FakeOrchestrator:
        audit = FakeAudit()

        def prepare(self, batch):
            return batch

        def analyse(self, batch, _ctx):
            return {
                "turbine_id": batch.turbine_id,
                "timestamp": int(batch.timestamps[-1]),
                "what": {"alarm": False, "fault": "healthy", "fault_confidence": 0.9,
                         "current_fused_score": 0.1},
            }

    monkeypatch.setattr(api, "_STATE", {
        "orchestrator": FakeOrchestrator(), "batch": None, "ctx": None,
        "fitted": True, "lock": threading.Lock(), "last": None,
        "feedback": [], "error": None,
    })
    server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(payload):
        request = Request(
            base + "/scada", data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    rows = [
        {"timestamp": 1_700_000_000 + i * 600, "wind_speed": 8.0,
         "power": 100.0, "rpm": 10.0}
        for i in range(2)
    ]
    try:
        status, result = post({"turbine_id": "WT-07", "rows": rows})
        assert status == 200
        assert result["writeback"]["tags"]["WTPM.TURBINE"] == "WT-07"
        assert result["writeback"]["tags"]["WTPM.ALARM"] == 0

        mixed = [dict(rows[0], turbine_id="WT-07"),
                 dict(rows[1], turbine_id="WT-08")]
        request = Request(
            base + "/scada", data=json.dumps({"rows": mixed}).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
        assert "one turbine" in json.loads(error.value.read())["error"]
        error.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_demo_analysis_includes_explanation(monkeypatch, tmp_path):
    # Exercise the actual default demo, without optional sibling repositories.
    # The full Docker image sets strict readiness; this test explicitly selects demo.
    monkeypatch.delenv("WTPM_REQUIRE_ALL_MODELS", raising=False)
    monkeypatch.setenv("WTPM_EXTERNAL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_STATE", {})
    api._bootstrap(days=6, seed=7)
    assert api._STATE["fitted"]
    result = api._STATE["orchestrator"].analyse(
        api._STATE["batch"], api._STATE["ctx"])
    assert result["contract_ok"]
    assert result["hermes"]["n_steps"] > 0
    assert result["why"]["narrative"]
    # The API must be able to serialize the whole result.
    json.dumps(result, default=api._json_default)
