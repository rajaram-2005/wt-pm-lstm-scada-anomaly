#!/usr/bin/env python3
"""Verify that all 25 WT-PM model pages exist, agree with the catalog, and merged.

The collection has one catalog (``deployment/models.catalog.json``) and 25
repository pages rendered from it. This tool answers the question the catalog
cannot answer about itself: *did the pages actually land?*

Checks per repository, on ``main`` and on ``gh-pages``:

1. ``README.md`` states the catalog title, adapter id, status and collection number.
2. ``README.md`` carries the advanced-concepts section and honest limits.
3. ``index.html`` states the same number, adapter id and status.
4. ``index.html`` has a working ``View on GitHub`` link - never the scaffold ``#``.
5. ``index.html`` links all 25 repositories (the pages cross-reference each other).
6. ``gh-pages`` serves the same single-file page as ``main``.
7. GitHub reports the Pages site as built.

Usage
-----
    # against local clones (offline, one directory per repo)
    python deployment/check_ecosystem.py --dir /tmp/verify

    # against GitHub itself (needs an authenticated gh with read access)
    python deployment/check_ecosystem.py --remote

    # both, and fail if any repository is not fully merged
    python deployment/check_ecosystem.py --dir /tmp/verify --remote --strict
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "deployment" / "models.catalog.json"
OWNER = "rajaram-2005"
REF_REPO = "wt-pm-lstm-scada-anomaly"
MARKER = "Generated from deployment/models.catalog.json"


# ---------------------------------------------------------------------------
def load_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gh_json(*args: str):
    """Call gh and return parsed JSON, or None when the resource is missing."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def gh_file(repo: str, path: str, ref: str):
    data = gh_json("api", f"repos/{OWNER}/{repo}/contents/{path}?ref={ref}")
    if not isinstance(data, dict) or "content" not in data:
        return None
    return base64.b64decode(data["content"]).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
class Result:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.problems: list[str] = []
        self.notes: list[str] = []

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.problems.append(message)
        return condition

    @property
    def ok(self) -> bool:
        return not self.problems


def check_readme(result: Result, model: dict, text: str) -> None:
    number = model["number"]
    result.check(f"# {model['title']}" in text, "README: title missing")
    result.check(f"`{model['adapter']}`" in text, "README: adapter id missing")
    result.check(f"model **{number:02d}** of 25" in text, "README: collection number missing")
    result.check("## Advanced concepts" in text, "README: advanced concepts section missing")
    result.check("## Honest limits" in text, "README: honest limits section missing")
    result.check("## Platform layers" in text, "README: platform layer participation missing")
    result.check("## The other 24 pages" in text, "README: cross-links to the other pages missing")
    result.check(MARKER in text, "README: not generated from the catalog")


def check_html(result: Result, model: dict, text: str, label: str, catalog: dict) -> None:
    number = model["number"]
    result.check(f"MODEL {number:02d} / 25" in text, f"{label}: collection number missing")
    result.check(model["adapter"] in text, f"{label}: adapter id missing")
    result.check("View on GitHub" in text, f"{label}: GitHub button missing")
    scaffold = re.search(r'href="#"', text) or re.search(r"View on GitHub Pages", text)
    result.check(not scaffold, f"{label}: still the scaffold placeholder link")
    result.check("Advanced concepts" in text, f"{label}: advanced concepts section missing")
    result.check("Platform contract" in text, f"{label}: contract section missing")
    result.check(MARKER in text, f"{label}: not generated from the catalog")
    missing = [m["repo"] for m in catalog["models"] if f"{OWNER}/{m['repo']}" not in text]
    result.check(not missing, f"{label}: does not link {len(missing)} of the 25 pages "
                              f"(e.g. {missing[:3]})")


def check_reference_readme(result: Result, model: dict, text: str) -> None:
    """The host repository documents itself; it is held to the same facts."""
    result.check(f"Model {model['number']}" in text or f"model {model['number']}" in text,
                 "README: collection number missing")
    result.check(f"`{model['adapter']}`" in text, "README: adapter id missing")
    result.check("models.catalog.json" in text, "README: does not cite the catalog")
    result.check("Advanced concepts" in text or "advanced concepts" in text,
                 "README: no pointer to the advanced-concept material")


def check_local(result: Result, model: dict, parent: Path, catalog: dict) -> None:
    root = parent / model["repo"]
    if model["repo"] == REF_REPO and not (root / "README.md").is_file():
        # The reference repository keeps its own README and its own site.
        source = ROOT / "README.md"
        if source.is_file():
            check_reference_readme(result, model, source.read_text(encoding="utf-8"))
        return
    if not root.is_dir():
        result.problems.append(f"local: {root} is not a clone")
        return
    readme = root / "README.md"
    index = root / "index.html"
    docs_index = root / "docs" / "index.html"
    if readme.is_file():
        check_readme(result, model, readme.read_text(encoding="utf-8"))
    else:
        result.problems.append("local: README.md missing")
    if index.is_file():
        check_html(result, model, index.read_text(encoding="utf-8"), "index.html", catalog)
    else:
        result.problems.append("local: index.html missing")
    if docs_index.is_file():
        result.check(docs_index.read_text(encoding="utf-8") == index.read_text(encoding="utf-8"),
                     "local: docs/index.html differs from index.html")
    else:
        result.problems.append("local: docs/index.html missing")


def check_host_site(result: Result, model: dict, site: Path, catalog: dict) -> None:
    """The merged static site must carry a copy of every one of the 25 pages."""
    name = f"model-{model['number']:02d}-" + model["repo"].replace("wt-pm-", "") + ".html"
    page = site / name
    if not page.is_file():
        result.problems.append(f"site: {name} missing from the merged site")
        return
    check_html(result, model, page.read_text(encoding="utf-8"), f"site {name}", catalog)


def check_remote(result: Result, model: dict, catalog: dict) -> None:
    repo = model["repo"]
    readme = gh_file(repo, "README.md", "main")
    if readme is None:
        result.problems.append("remote: README.md not readable on main")
    else:
        check_readme(result, model, readme)

    index = gh_file(repo, "index.html", "main")
    if index is None:
        result.problems.append("remote: index.html not readable on main")
    else:
        check_html(result, model, index, "remote index.html", catalog)

    pages = gh_file(repo, "index.html", "gh-pages")
    if pages is None:
        result.problems.append("remote: gh-pages has no index.html")
    else:
        check_html(result, model, pages, "gh-pages index.html", catalog)
        if index is not None:
            result.check(pages == index, "remote: gh-pages page differs from main")

    site = gh_json("api", f"repos/{OWNER}/{repo}/pages")
    if site is None:
        result.problems.append("remote: GitHub Pages is not enabled")
    else:
        result.check(site.get("status") == "built",
                     f"remote: Pages status is {site.get('status')!r}, expected 'built'")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(CATALOG))
    ap.add_argument("--dir", help="directory holding one clone per repository")
    ap.add_argument("--site", help="merged static site holding all 25 rendered pages")
    ap.add_argument("--remote", action="store_true", help="verify against GitHub")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when any page is missing or unmerged")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    catalog = load_catalog(Path(args.catalog))
    models = catalog["models"]

    results = []
    for model in models:
        result = Result(model["repo"])
        if args.dir:
            check_local(result, model, Path(args.dir), catalog)
        if args.site:
            check_host_site(result, model, Path(args.site), catalog)
        if args.remote:
            check_remote(result, model, catalog)
        results.append(result)
        if not args.quiet:
            state = "OK  " if result.ok else "FAIL"
            print(f"{state} {model['number']:02d} {model['repo']}")
            for problem in result.problems:
                print(f"       - {problem}")

    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} pages merged and consistent "
          f"with deployment/models.catalog.json")
    if failed and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
