#!/usr/bin/env python3
"""Render the 25 WT-PM model pages from the single canonical catalog.

``deployment/models.catalog.json`` is the only place where a model's number,
adapter id, status, contract and advanced-concept notes are written down.
Everything a reader sees on a repository page - README.md, index.html and
docs/index.html - is rendered from it, so the 25 pages cannot drift apart.

Usage
-----
    python deployment/build_model_pages.py --out build/pages          # all 25
    python deployment/build_model_pages.py --repo wt-pm-gru-scada-telemetry
    python deployment/build_model_pages.py --site build/site          # merged site
    python deployment/build_model_pages.py --check                    # catalog sanity

Layout written per repository (identical for every model)::

    <repo>/README.md
    <repo>/index.html
    <repo>/docs/index.html

The generated files are self-contained: no external CSS, JS, fonts or images,
so a ``gh-pages`` branch can hold the landing page as a single file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "deployment" / "models.catalog.json"

GH = "https://github.com/rajaram-2005/"
REF_REPO = "wt-pm-lstm-scada-anomaly"
ARCH_URL = f"{GH}{REF_REPO}/blob/main/ARCHITECTURE.md"
MODELS_URL = f"{GH}{REF_REPO}/blob/main/platform/docs/MODELS.md"
ECOSYSTEM_URL = "https://rajaram-2005.github.io/wt-pm-lstm-scada-anomaly/ecosystem.html"


# ---------------------------------------------------------------------------
# catalog helpers
# ---------------------------------------------------------------------------
def load_catalog(path: Path = CATALOG_PATH) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    validate(catalog)
    return catalog


def validate(catalog: dict) -> None:
    """Fail loudly rather than emit 25 pages that disagree with each other."""
    models = catalog["models"]
    numbers = [m["number"] for m in models]
    if numbers != list(range(1, 26)):
        raise ValueError(f"collection numbering must be 1..25, got {numbers}")
    for key in ("repo", "adapter", "title"):
        dupes = {m[key] for m in models if [x[key] for x in models].count(m[key]) > 1}
        if dupes:
            raise ValueError(f"duplicate {key}: {sorted(dupes)}")
    known = set(numbers)
    for a, b, _ in catalog["edges"]:
        if a not in known or b not in known:
            raise ValueError(f"edge references unknown model: {a}->{b}")
    for layer in catalog["advanced_layers"]:
        unknown = set(layer["models"]) - known
        if unknown:
            raise ValueError(f"layer {layer['key']} references {sorted(unknown)}")
    for m in models:
        if m["category"] not in catalog["categories"]:
            raise ValueError(f"{m['repo']}: unknown category {m['category']}")
        if not m.get("advanced"):
            raise ValueError(f"{m['repo']}: no advanced concepts")
        if not m.get("layers"):
            raise ValueError(f"{m['repo']}: no platform layers")


def by_number(catalog: dict) -> dict:
    return {m["number"]: m for m in catalog["models"]}


def relations(catalog: dict, number: int):
    """Return (incoming, outgoing) edge lists for one model."""
    out, inn = [], []
    index = by_number(catalog)
    for a, b, label in catalog["edges"]:
        if a == number:
            out.append((index[b], label))
        if b == number:
            inn.append((index[a], label))
    return inn, out


#: layer names used by the per-model pages, mapped onto the catalog definitions
LAYER_ALIASES = {
    "fusion": "Fusion", "uncertainty": "Uncertainty", "drift": "Drift",
    "xai": "XAI", "physics": "Physics", "safety": "Safety gate",
    "protocol": "Protocol", "hermes": "Hermes",
}


def layers_of(catalog: dict, number: int) -> list:
    """The layer-participation rows written for this model (name + detail)."""
    model = by_number(catalog)[number]
    return [{"name": row["layer"], "detail": row["detail"]} for row in model["layers"]]


def layer_definitions(catalog: dict) -> dict:
    """Catalog layer key -> full definition, for the merged site page."""
    return {L["key"]: L for L in catalog["advanced_layers"]}


def related_numbers(catalog: dict, number: int) -> list:
    rel = []
    for a, b, _ in catalog["edges"]:
        for other in (b if a == number else a if b == number else None,):
            if other is not None and other not in rel:
                rel.append(other)
    return sorted(rel)


def label_of(model: dict) -> str:
    return f"{model['number']:02d} / {model['adapter']}"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def render_readme(catalog: dict, model: dict) -> str:
    number = model["number"]
    cat = catalog["categories"][model["category"]]
    io = model["io"]
    inn, out = relations(catalog, number)
    layers = layers_of(catalog, number)
    related = related_numbers(catalog, number)
    index = by_number(catalog)

    status_kind = model.get("status_kind", "integrated")
    badge = {"integrated": "integrated", "partial": "partial", "complete": "complete"}[status_kind]

    lines: list[str] = []
    lines.append(f"# {model['title']}")
    lines.append("")
    lines.append(f"> **Category:** {cat['name']}  ")
    lines.append(f"> **Platform adapter:** `{model['adapter']}`  ")
    lines.append(f"> **Status:** {model['status']}  ")
    lines.append(f"> **Collection position:** model **{number:02d}** of 25 (alphabetical)")
    lines.append("")
    lines.append("<!-- Generated from deployment/models.catalog.json - edit the catalog, not this file. -->")
    lines.append("")
    lines.append(model["summary"])
    lines.append("")

    lines.append("## How this page fits the collection")
    lines.append("")
    lines.append(
        "The WT-PM collection numbers its 25 repositories **alphabetically 01-25** - that is the "
        f"number on this page - and gives each one a **task-based adapter id** (`{model['adapter']}`) "
        "used by the platform registry, the `wt-pm` CLI and the fusion engine. Both are printed on "
        "every page so the ecosystem graph and the registry can be read against each other."
    )
    lines.append("")
    lines.append(f"- Repository: <{GH + model['repo']}>")
    lines.append(f"- Landing page: <https://rajaram-2005.github.io/{model['repo']}/>")
    lines.append(f"- Ecosystem graph: <{ECOSYSTEM_URL}>")
    lines.append(f"- Platform architecture: <{ARCH_URL}>")
    lines.append(f"- Per-model table: <{MODELS_URL}>")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    for entry_point in model["entry_points"]:
        lines.append(f"- `{entry_point}`" if not entry_point.startswith("python") else f"- {entry_point}")
    lines.append("")
    lines.append("| Block | Detail |")
    lines.append("|---|---|")
    for row in model["architecture"]:
        lines.append(f"| {row['block']} | {row['detail']} |")
    lines.append("")

    lines.append("## Advanced concepts")
    lines.append("")
    lines.append(
        "The ideas this model actually leans on - stated with their failure modes, not as marketing."
    )
    lines.append("")
    for item in model["advanced"]:
        lines.append(f"### {item['concept']}")
        lines.append("")
        lines.append(item["detail"])
        lines.append("")

    lines.append("## Platform contract")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Input (adapter view) | {io['input']} |")
    lines.append(f"| Output (`WTDataSchema`) | {io['output']} |")
    lines.append(f"| Record format | `{io['record']}` |")
    lines.append(f"| Typical latency | {io['latency_ms']} ms |")
    lines.append(f"| Deployment targets | {', '.join(io['deployment'])} |")
    lines.append(f"| Resource requirements | {', '.join(io['deps']) or 'none'} |")
    lines.append(f"| Fallback | {('`' + io['fallback'] + '`') if io.get('fallback') else 'none - this model is a leaf'} |")
    lines.append(f"| Subsystem focus | {io['subsystem'] or 'whole turbine'} |")
    lines.append(f"| Hermes tool | `{model.get('hermes', 'n/a')}` |")
    lines.append("")
    lines.append(
        "Records follow the shared `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and "
        "`model_id` are required; every other field is optional, so a model fills in only what it "
        "honestly produces and the fusion layer can tell absence from a zero."
    )
    lines.append("")

    lines.append("## Platform layers")
    lines.append("")
    lines.append("| Layer | What this model does there |")
    lines.append("|---|---|")
    for layer in layers:
        lines.append(f"| **{layer['name']}** | {layer['detail']} |")
    lines.append("")
    lines.append(
        "Layers are described once, in the catalog, and referenced by every model that participates "
        f"in them: fusion, calibrated uncertainty, drift and data trust, explainability, physics "
        f"constraints, the safety gate, the evaluation protocol and the Hermes agent."
    )
    lines.append("")

    lines.append("## How it connects to the other 24")
    lines.append("")
    if inn or out:
        if inn:
            lines.append("**Receives from**")
            lines.append("")
            for other, label in inn:
                lines.append(f"- **{other['number']:02d}. {other['title']}** (`{other['adapter']}`) - {label}")
            lines.append("")
        if out:
            lines.append("**Feeds**")
            lines.append("")
            for other, label in out:
                lines.append(f"- **{other['number']:02d}. {other['title']}** (`{other['adapter']}`) - {label}")
            lines.append("")
    else:
        lines.append(
            "No direct functional edge: this model is coupled to the collection through the shared "
            "contract (schema, registry, evaluation protocol) rather than through a data dependency."
        )
        lines.append("")
    if related:
        lines.append("Related pages: " + ", ".join(
            f"[{index[n]['number']:02d} {index[n]['short']}]({GH + index[n]['repo']})" for n in related
        ))
    lines.append("")

    lines.append("## Standalone setup")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("python model.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "This repository stays independently runnable. The platform wraps the original `model.py` "
        "through its adapter - it does not replace, patch or fork this code."
    )
    lines.append("")

    lines.append("## Run it inside the platform")
    lines.append("")
    lines.append("```bash")
    lines.append('pip install "git+https://github.com/rajaram-2005/' + REF_REPO + '.git#subdirectory=platform"')
    lines.append("wt-pm fetch-models --dir ./external      # clone this repo and its siblings")
    lines.append("export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py")
    lines.append(f"wt-pm inspect                            # {model['adapter']} should report available")
    lines.append("wt-pm serve --port 8100                  # HTTP API + command centre")
    lines.append("```")
    lines.append("")

    lines.append("## Honest limits")
    lines.append("")
    for limit in model["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append(
        "Fidelity: rung 1 of the fidelity ladder - simulator data, surrogate waveforms where stated, "
        "proxy RUL targets and simulator fault labels. Results are comparable **within** this "
        "collection and are not field-validated turbine control."
    )
    lines.append("")

    lines.append("## The other 24 pages")
    lines.append("")
    for m in catalog["models"]:
        if m["number"] == number:
            continue
        marker = " <- this repo" if m["number"] == number else ""
        lines.append(f"- {m['number']:02d}. [{m['title']}]({GH + m['repo']}) - `{m['adapter']}`{marker}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Part of the [WT-PM collection]({GH}{REF_REPO}) - 25 models, one contract. "
                 "Generated from `deployment/models.catalog.json`.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML landing page (self-contained, no external assets)
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#05080c; --panel:#0b1216; --ink:#e8f4f2; --muted:#8aa4a8; --line:rgba(46,230,200,.18);
  --teal:#2ee6c8; --amber:#f0b429;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Segoe UI",system-ui,-apple-system,sans-serif;line-height:1.6}
a{color:var(--teal);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 64px}
header{border-bottom:1px solid var(--line);padding:26px 0 22px;margin-bottom:34px}
.crumbs{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:12.5px;color:var(--muted)}
.pill{border:1px solid var(--line);border-radius:999px;padding:3px 11px;font-weight:700;letter-spacing:.06em}
.pill.num{background:var(--teal);color:#04231e;border-color:var(--teal)}
.pill.adapter{color:var(--teal)}
.pill.status{color:var(--amber);border-color:rgba(240,180,41,.4)}
h1{font-size:clamp(28px,4vw,42px);margin:18px 0 10px;line-height:1.12}
.lede{color:#c9ddd9;font-size:17px;max-width:52rem}
h2{font-size:21px;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}
p{color:#b7d0cc}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14.5px}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;width:210px}
td{color:#c9ddd9}
code{background:#101a20;border:1px solid var(--line);border-radius:6px;padding:1px 6px;color:#7dd3fc;font-size:13.5px}
pre{background:#0b1216;border:1px solid var(--line);border-radius:12px;padding:16px;overflow:auto;color:#cde}
.card{border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:12px 0;background:var(--panel)}
.card h3{margin-top:0;color:var(--teal)}
ul{color:#b7d0cc}
li{margin:5px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:14px 0}
.fleet a{display:block;border:1px solid var(--line);border-radius:11px;padding:10px 12px;font-size:13.5px;color:#c9ddd9}
.fleet a:hover{border-color:var(--teal);text-decoration:none}
.fleet a b{display:block;color:var(--ink);font-size:14px;margin-bottom:2px}
.fleet a span{color:var(--muted);font-size:11.5px}
.fleet a.here{outline:2px solid var(--teal);outline-offset:-2px}
.edges li b{color:var(--ink)}
.limits li{color:#e0b7b7}
footer{margin-top:56px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}
.note{color:var(--muted);font-size:13.5px}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 6px}
.btn{border:1px solid var(--teal);border-radius:999px;padding:9px 16px;font-weight:650;color:#04231e;background:var(--teal)}
.btn.ghost{background:transparent;color:var(--ink)}
"""


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_html(catalog: dict, model: dict) -> str:
    number = model["number"]
    cat = catalog["categories"][model["category"]]
    io = model["io"]
    index = by_number(catalog)
    inn, out = relations(catalog, number)
    layers = layers_of(catalog, number)

    parts: list[str] = []
    add = parts.append

    add("<!DOCTYPE html>")
    add("<!-- Generated from deployment/models.catalog.json - edit the catalog, not this file. -->")
    add('<html lang="en"><head><meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add(f"<title>{esc(model['title'])} - WT-PM model {number:02d} of 25</title>")
    add(f"<meta name=\"description\" content=\"{esc(model['summary'][:150])}\">")
    add(f"<style>{CSS}</style></head><body><div class=\"wrap\">")

    add('<header><div class="crumbs">')
    add('<a href="https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly">WT-PM</a>')
    add(f'<span>collection of 25 models</span>')
    add(f'<span class="pill num">MODEL {number:02d} / 25</span>')
    add(f'<span class="pill adapter">{esc(model["adapter"])}</span>')
    add(f'<span class="pill status">{esc(model["status"])}</span>')
    add(f'<span class="pill" style="color:{cat["color"]};border-color:{cat["color"]}55">{esc(cat["name"])}</span>')
    add("</div>")
    add(f"<h1>{esc(model['title'])}</h1>")
    add(f'<p class="lede">{esc(model["summary"])}</p>')
    add('<div class="cta">')
    add(f'<a class="btn" href="{GH}{model["repo"]}">View on GitHub</a>')
    add(f'<a class="btn ghost" href="{ECOSYSTEM_URL}">25-model ecosystem graph</a>')
    add(f'<a class="btn ghost" href="{ARCH_URL}">Platform architecture</a>')
    add("</div>")
    add("</header>")

    add('<p class="note">This page is generated from <code>deployment/models.catalog.json</code> in '
        'the reference repository, together with the other 24 model pages, so all 25 agree on '
        'numbering, contract and status.</p>')

    add("<h2>Model</h2>")
    add("<ul>")
    for entry in model["entry_points"]:
        add(f"<li><code>{esc(entry.split(' - ')[0])}</code>"
            + (f" - {esc(entry.split(' - ', 1)[1])}" if " - " in entry else "") + "</li>")
    add("</ul>")
    add("<table>")
    for row in model["architecture"]:
        add(f"<tr><th>{esc(row['block'])}</th><td>{esc(row['detail'])}</td></tr>")
    add("</table>")

    add("<h2>Advanced concepts</h2>")
    add('<p class="note">The ideas this model actually leans on, each stated with its failure mode.</p>')
    for item in model["advanced"]:
        add('<div class="card">')
        add(f"<h3>{esc(item['concept'])}</h3>")
        add(f"<p>{esc(item['detail'])}</p>")
        add("</div>")

    add("<h2>Platform contract</h2>")
    add("<table>")
    add(f"<tr><th>Input (adapter view)</th><td>{esc(io['input'])}</td></tr>")
    add(f"<tr><th>Output</th><td>{esc(io['output'])}</td></tr>")
    add(f"<tr><th>Record format</th><td><code>{esc(io['record'])}</code></td></tr>")
    add(f"<tr><th>Typical latency</th><td>{io['latency_ms']} ms</td></tr>")
    add(f"<tr><th>Deployment targets</th><td>{esc(', '.join(io['deployment']))}</td></tr>")
    add(f"<tr><th>Requirements</th><td>{esc(', '.join(io['deps']) or 'none')}</td></tr>")
    add(f"<tr><th>Fallback</th><td>{esc(io['fallback']) if io.get('fallback') else 'none - leaf model'}</td></tr>")
    add(f"<tr><th>Subsystem focus</th><td>{esc(io['subsystem'] or 'whole turbine')}</td></tr>")
    add(f"<tr><th>Hermes tool</th><td><code>{esc(model.get('hermes', 'n/a'))}</code></td></tr>")
    add("</table>")
    add('<p class="note">Records follow the shared <code>WTDataSchema</code> '
        '(<code>wt-pm.platform.v1</code>). Only <code>timestamp</code>, <code>turbine_id</code> and '
        '<code>model_id</code> are required; absence is distinguishable from a zero.</p>')

    add("<h2>Platform layers</h2>")
    add("<table>")
    for layer in layers:
        add(f"<tr><th>{esc(layer['name'])}</th><td>{esc(layer['detail'])}</td></tr>")
    add("</table>")

    add("<h2>How it connects to the other 24</h2>")
    if inn or out:
        add('<ul class="edges">')
        for other, label in inn:
            add(f"<li><b>{other['number']:02d}. {esc(other['title'])}</b> "
                f"<span class=\"note\">({esc(other['adapter'])})</span> &rarr; {esc(label)}</li>")
        for other, label in out:
            add(f"<li>&rarr; <b>{other['number']:02d}. {esc(other['title'])}</b> "
                f"<span class=\"note\">({esc(other['adapter'])})</span> - {esc(label)}</li>")
        add("</ul>")
    else:
        add('<p class="note">No direct functional edge: coupled to the collection through the shared '
            "contract rather than a data dependency.</p>")

    add("<h2>Honest limits</h2><ul class=\"limits\">")
    for limit in model["limits"]:
        add(f"<li>{esc(limit)}</li>")
    add("</ul>")
    add('<p class="note">Fidelity: rung 1 of the fidelity ladder - simulator data, surrogate waveforms '
        'where stated, proxy RUL targets, simulator fault labels. Comparable within this collection, '
        'not field-validated turbine control.</p>')

    add("<h2>Run it</h2>")
    add("<pre>pip install -r requirements.txt\npython model.py\n\n"
        "# inside the unified platform\n"
        'pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"\n'
        "wt-pm fetch-models --dir ./external\n"
        "export WTPM_EXTERNAL_DIR=$PWD/external\n"
        f"wt-pm inspect      # {esc(model['adapter'])} should report available\n"
        "wt-pm serve --port 8100</pre>")

    add("<h2>All 25 pages</h2>")
    add('<p class="note">Every repository in the collection, with the number printed on its own page and '
        'the adapter id the registry uses.</p>')
    add('<div class="grid fleet">')
    for m in catalog["models"]:
        here = " here" if m["number"] == number else ""
        add(f'<a class="{here.strip()}" href="{GH}{m["repo"]}"><b>{m["number"]:02d} &middot; {esc(m["title"])}</b>'
            f'<span>{esc(m["adapter"])} &middot; {esc(catalog["categories"][m["category"]]["name"])}</span></a>')
    add("</div>")

    add("<footer>")
    add(f'<p>Model {number:02d} of 25 &middot; adapter <code>{esc(model["adapter"])}</code> &middot; '
        f'{esc(cat["name"])} &middot; status {esc(model["status"])}</p>')
    add('<p><a href="https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md">'
        "Per-model table</a> &middot; "
        f'<a href="{ECOSYSTEM_URL}">Ecosystem graph</a> &middot; '
        '<a href="https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly">Reference repository</a></p>')
    add('<p>Generated from <code>deployment/models.catalog.json</code>. '
        "Research demo - not field-validated turbine control.</p>")
    add("</footer>")

    add("</div></body></html>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# merged site page - the 25 pages in one view
# ---------------------------------------------------------------------------
def page_name(model: dict) -> str:
    """Stable file name for a model's page inside the merged static site."""
    slug = model["repo"].replace("wt-pm-", "")
    return f"model-{model['number']:02d}-{slug}.html"


def render_site_index(catalog: dict) -> str:
    index = by_number(catalog)
    out: list[str] = []
    add = out.append
    add("<!DOCTYPE html>")
    add("<!-- Generated from deployment/models.catalog.json - edit the catalog, not this file. -->")
    add('<html lang="en"><head><meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add("<title>WT-PM - all 25 model pages</title>")
    add(f"<style>{CSS}</style></head><body><div class=\"wrap\">")
    add('<header><div class="crumbs"><a href="index.html">WT-PM</a><span>static site</span>'
        '<span class="pill">25 MODEL PAGES</span></div>')
    add("<h1>All 25 model pages</h1>")
    add('<p class="lede">One catalog, one contract, twenty-five pages. Each card links to a repository '
        "page that states the same numbering, the same record format and the same honest limits.</p>")
    add("</header>")

    for key, cat in catalog["categories"].items():
        models = [m for m in catalog["models"] if m["category"] == key]
        add(f'<h2 style="border-color:{cat["color"]}55">{esc(cat["name"])} '
            f'<span class="note">({len(models)} models)</span></h2>')
        add(f'<p class="note">{esc(cat["note"])}</p>')
        add('<div class="grid fleet">')
        for m in models:
            add(f'<a href="{page_name(m)}"><b>{m["number"]:02d} &middot; {esc(m["title"])}</b>'
                f'<span>{esc(m["adapter"])} &middot; {esc(m["status"])}</span></a>')
        add("</div>")

    add("<h2>Platform layers</h2>")
    add('<p class="note">Every model page names the layers it participates in. The layers themselves '
        "are defined once, in the catalog.</p>")
    add("<table>")
    for layer in catalog["advanced_layers"]:
        names = ", ".join(f"{index[n]['number']:02d}" for n in layer["models"])
        add(f"<tr><th>{esc(layer['name'])}</th><td>{esc(layer['what'])}<br>"
            f'<span class="note">{esc(layer["how"])}</span><br>'
            f'<span class="note">models: {names}</span></td></tr>')
    add("</table>")

    add("<h2>All 25 pages on this site</h2>")
    add('<p class="note">The same pages that are published as each repository\'s landing page, '
        "rendered here from the same catalog so they can be read side by side.</p>")
    add("<table>")
    for m in catalog["models"]:
        add(f'<tr><th><a href="{page_name(m)}">{m["number"]:02d} &middot; {esc(m["title"])}</a></th>'
            f'<td><a href="{GH}{m["repo"]}">{esc(m["repo"])}</a><br>'
            f'<span class="note"><code>{esc(m["adapter"])}</code> &middot; {esc(m["status"])}</span></td></tr>')
    add("</table>")

    add("<h2>Concept index</h2>")
    add('<p class="note">Every advanced concept named anywhere in the collection, and the pages that '
        "explain it.</p>")
    add("<table>")
    for concept, nums in catalog["concept_index"].items():
        links = ", ".join(f'<a href="{GH}{index[n]["repo"]}">{n:02d}</a>' for n in nums)
        add(f"<tr><th>{esc(concept)}</th><td>{links}</td></tr>")
    add("</table>")

    add("<footer><p>Generated from <code>deployment/models.catalog.json</code> in "
        '<a href="https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly">wt-pm-lstm-scada-anomaly</a>. '
        "Research demo - not field-validated turbine control.</p></footer>")
    add("</div></body></html>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def write_repo(out: Path, catalog: dict, model: dict, with_readme: bool = True) -> list:
    root = out / model["repo"]
    (root / "docs").mkdir(parents=True, exist_ok=True)
    written = []
    html = render_html(catalog, model)
    if with_readme:
        (root / "README.md").write_text(render_readme(catalog, model), encoding="utf-8")
        written.append(root / "README.md")
    (root / "index.html").write_text(html, encoding="utf-8")
    (root / "docs" / "index.html").write_text(html, encoding="utf-8")
    written += [root / "index.html", root / "docs" / "index.html"]
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(CATALOG_PATH))
    ap.add_argument("--out", help="directory for one rendered repository per model")
    ap.add_argument("--repo", help="render only this repository")
    ap.add_argument("--site", help="write the merged all-25 site page to this directory")
    ap.add_argument("--check", action="store_true", help="validate the catalog and exit")
    args = ap.parse_args(argv)

    catalog = load_catalog(Path(args.catalog))
    models = catalog["models"]

    if args.check:
        print(f"catalog ok: {len(models)} models, {len(catalog['edges'])} edges, "
              f"{len(catalog['advanced_layers'])} layers, {len(catalog['concept_index'])} concepts")
        return 0

    if args.repo:
        match = [m for m in models if m["repo"] == args.repo]
        if not match:
            raise SystemExit(f"unknown repository: {args.repo}")
        models = match
    if args.out:
        out = Path(args.out)
        count = 0
        for model in models:
            # Never overwrite the reference repository's own README.
            count += len(write_repo(out, catalog, model, with_readme=model["repo"] != REF_REPO))
        print(f"rendered {count} files for {len(models)} repositories under {out}")
    if args.site:
        site = Path(args.site)
        site.mkdir(parents=True, exist_ok=True)
        (site / "all-models.html").write_text(render_site_index(catalog), encoding="utf-8")
        for model in catalog["models"]:
            (site / page_name(model)).write_text(render_html(catalog, model), encoding="utf-8")
        print(f"wrote 1 index + {len(catalog['models'])} model pages to {site}")
    if not (args.out or args.site):
        print(f"catalog has {len(models)} models; nothing to do (use --out or --site)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
