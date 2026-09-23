"""Cloud deployment contract: PORT and a real readiness probe."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

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


def test_demo_analysis_includes_explanation(monkeypatch, tmp_path):
    # Exercise the actual default demo, without optional sibling repositories.
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
