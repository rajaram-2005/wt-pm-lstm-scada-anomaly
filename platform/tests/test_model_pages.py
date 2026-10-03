"""The 25 model pages, the catalog and the live registry must agree.

These tests are the guard rail for "all 25 pages merge correctly": one catalog
renders 25 pages, and every catalog row must correspond to a real adapter in the
platform's ModelRegistry. If someone renames an adapter, renumbers a repository
or edits a page by hand, this fails before the pages can drift apart.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "deployment" / "models.catalog.json"
LAUNCH = ROOT / "docs" / "launch"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    return _load("build_model_pages", ROOT / "deployment" / "build_model_pages.py")


@pytest.fixture(scope="module")
def checker():
    return _load("check_ecosystem", ROOT / "deployment" / "check_ecosystem.py")


# ---------------------------------------------------------------------------
# catalog structure
# ---------------------------------------------------------------------------
def test_catalog_numbers_all_25_models(catalog):
    assert [m["number"] for m in catalog["models"]] == list(range(1, 26))
    assert len({m["repo"] for m in catalog["models"]}) == 25
    assert len({m["adapter"] for m in catalog["models"]}) == 25


def test_every_model_has_advanced_content(catalog):
    for model in catalog["models"]:
        assert len(model["advanced"]) >= 3, model["repo"]
        assert model["layers"], model["repo"]
        assert model["limits"], model["repo"]
        assert model["architecture"], model["repo"]
        assert model["io"]["record"].startswith("wt-pm.platform.v1")


def test_edges_and_layers_reference_real_models(catalog):
    numbers = {m["number"] for m in catalog["models"]}
    assert len(catalog["edges"]) >= 25
    for a, b, label in catalog["edges"]:
        assert a in numbers and b in numbers and label
        assert a != b
    for layer in catalog["advanced_layers"]:
        assert set(layer["models"]) <= numbers
        assert layer["what"] and layer["how"] and layer["why"]


def test_concept_index_covers_every_model(catalog):
    indexed = {n for nums in catalog["concept_index"].values() for n in nums}
    assert indexed == {m["number"] for m in catalog["models"]}
    assert len(catalog["concept_index"]) >= 50


def test_catalog_references_the_host_repository(catalog):
    host = [m for m in catalog["models"] if m["repo"] == catalog["reference_repo"]]
    assert len(host) == 1
    assert host[0]["adapter"] == "m05-lstm-scada-anomaly"
    assert host[0]["number"] == 13
    assert host[0]["hub"] is True


# ---------------------------------------------------------------------------
# catalog <-> live platform registry
# ---------------------------------------------------------------------------
def test_catalog_matches_the_platform_registry(catalog):
    """The adapter ids on the pages are the ids the platform actually registers."""
    from wtpm_platform.orchestrator import build_default_registry

    registry = build_default_registry()
    specs = {mid: registry.get(mid).spec for mid in registry.ids()}
    assert len(specs) == 25

    for model in catalog["models"]:
        spec = specs.get(model["adapter"])
        assert spec is not None, f"{model['adapter']} is not registered by the platform"
        assert spec.repository == model["repo"], model["adapter"]
        assert model["io"]["latency_ms"] == spec.typical_latency_ms
        assert model["io"]["fallback"] == spec.fallback
        assert set(model["io"]["deployment"]) == {d.value for d in spec.deployment_targets}
        assert set(model["io"]["deps"]) <= set(spec.resource_requirements) | {"torch-geometric (optional: the repo ships a fallback)", "micromlgen (for C++ export)", "wt_pm_lstm"}


def test_catalog_repositories_exist_upstream(catalog):
    """Names must match the real GitHub repositories (checked in CI by gh, offline here)."""
    for model in catalog["models"]:
        assert re.fullmatch(r"wt-pm-[a-z0-9-]+", model["repo"])
    assert len({m["repo"] for m in catalog["models"]}) == 25


# ---------------------------------------------------------------------------
# rendered pages
# ---------------------------------------------------------------------------
def test_pages_render_and_verify(catalog, builder, checker, tmp_path):
    """Render all 25 pages and run the merge checker over them."""
    out = tmp_path / "pages"
    for model in catalog["models"]:
        builder.write_repo(out, catalog, model, with_readme=model["repo"] != catalog["reference_repo"])

    results = []
    for model in catalog["models"]:
        if model["repo"] == catalog["reference_repo"]:
            continue  # the host repository documents itself
        results.append(checker.Result(model["repo"]))
        checker.check_local(results[-1], model, out, catalog)
        results[-1].check(
            (out / model["repo"] / "index.html").read_text(encoding="utf-8")
            == (out / model["repo"] / "docs" / "index.html").read_text(encoding="utf-8"),
            "docs/index.html must equal index.html",
        )

    problems = {r.repo: r.problems for r in results if not r.ok}
    assert not problems, problems


def test_every_page_states_both_numbering_schemes(catalog, builder, tmp_path):
    for model in catalog["models"]:
        html = builder.render_html(catalog, model)
        assert f"MODEL {model['number']:02d} / 25" in html
        assert model["adapter"] in html
        readme = builder.render_readme(catalog, model)
        assert f"model **{model['number']:02d}** of 25" in readme
        assert f"`{model['adapter']}`" in readme


def test_pages_cross_link_all_25_repositories(catalog, builder):
    for model in catalog["models"]:
        html = builder.render_html(catalog, model)
        missing = [m["repo"] for m in catalog["models"] if m["repo"] not in html]
        assert not missing, f"{model['repo']} does not link {missing}"


# ---------------------------------------------------------------------------
# published static site
# ---------------------------------------------------------------------------
def test_site_ships_all_25_pages(catalog):
    assert LAUNCH.is_dir(), "docs/launch is missing"
    pages = sorted(LAUNCH.glob("model-*.html"))
    assert len(pages) == 25, [p.name for p in pages]
    text = (LAUNCH / "all-models.html").read_text(encoding="utf-8")
    for model in catalog["models"]:
        assert f"{model['number']:02d} &middot; {model['title']}" in text


def test_site_pages_match_the_catalog(catalog):
    for model in catalog["models"]:
        slug = model["repo"].replace("wt-pm-", "")
        page = LAUNCH / f"model-{model['number']:02d}-{slug}.html"
        html = page.read_text(encoding="utf-8")
        assert model["adapter"] in html
        assert f"MODEL {model['number']:02d} / 25" in html
        assert "deployment/models.catalog.json" in html


def test_ecosystem_graph_knows_every_model(catalog):
    html = (ROOT / "docs" / "ecosystem.html").read_text(encoding="utf-8")
    for model in catalog["models"]:
        assert f'"{model["repo"]}"' in html, model["repo"]
    for layer in catalog["advanced_layers"]:
        assert layer["name"] in html, layer["key"]
    assert "all-models.html" in html
