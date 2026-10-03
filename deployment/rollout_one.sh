#!/usr/bin/env bash
# Roll the advanced 25-page content out to one sibling repository:
# branch -> commit -> PR -> merge -> rebuild gh-pages -> verify.
#
# Usage: bash deployment/rollout_one.sh <repo> [clone-dir] [pages-dir]
set -euo pipefail

REPO="${1:?repo}"
CLONE="${2:-/tmp/roll/$REPO}"
PAGES="${3:-/tmp/pages}"
BRANCH="arena/01a1001d-advanced-pages"
API="rajaram-2005"

cd "$CLONE"
git checkout -q main
git pull -q --ff-only origin main
git branch -D "$BRANCH" >/dev/null 2>&1 || true
git checkout -q -b "$BRANCH"

mkdir -p docs
cp "$PAGES/$REPO/README.md" README.md
cp "$PAGES/$REPO/index.html" index.html
cp "$PAGES/$REPO/docs/index.html" docs/index.html

git add README.md index.html docs/index.html
git -c user.email="rajaram-2005@users.noreply.github.com" -c user.name="Rajaraman" \
    commit -q -m "Advanced-concepts page and platform wiring

Generated from deployment/models.catalog.json in the reference repository, so
this page agrees with the other 24 on collection numbering, adapter id, status,
contract and honest limits.

- README.md: collection position (both numbering schemes), architecture table,
  advanced concepts with failure modes, platform contract, platform-layer
  participation, cross-links to the other 24 pages, standalone + platform setup,
  honest limits.
- index.html and docs/index.html: the same content as a self-contained landing
  page - no external CSS/JS/assets - with a working View on GitHub link and
  links to all 25 repositories.
- model.py and requirements.txt untouched; the repository stays independently
  runnable."

git push -q -u origin "$BRANCH"

gh pr create --repo "$API/$REPO" --base main --head "$BRANCH" \
  --title "Advanced concepts on all 25 pages: model-level page for this repository" \
  --body "Part of the collection-wide page upgrade (\`arena/01a1001d-wt-pm-lstm-scada-anomaly\`).

Every one of the 25 repositories gets the same treatment, rendered from one
canonical catalog in \`wt-pm-lstm-scada-anomaly/deployment/models.catalog.json\`:

- **Collection numbering stated explicitly** - alphabetical 01-25 *and* the
  task-based platform adapter id, so the ecosystem graph and the registry can be
  read against each other.
- **Advanced concepts** per model (3-4 each, with failure modes).
- **Platform layers** the model participates in: fusion, calibrated uncertainty,
  drift and data trust, explainability, physics constraints, the safety gate,
  the evaluation protocol, the Hermes agent.
- **Cross-links to all 25 pages** plus the merged index.
- **Honest limits** and the fidelity rung.

\`model.py\` and \`requirements.txt\` are untouched." \
  >/dev/null 2>&1 || echo "(PR already exists)"

gh pr merge "$BRANCH" --repo "$API/$REPO" --squash --delete-branch \
  >/dev/null 2>&1 || gh pr merge "$BRANCH" --repo "$API/$REPO" --squash --admin --delete-branch

# Rebuild gh-pages as a clean single-file site.
TMP="$(mktemp -d)"
cp "$PAGES/$REPO/index.html" "$TMP/index.html"
( cd "$TMP"
  git init -q -b gh-pages
  git add index.html
  git -c user.email="rajaram-2005@users.noreply.github.com" -c user.name="Rajaraman" \
      commit -q -m "Publish landing page only (clean gh-pages)"
  git remote add origin "https://github.com/$API/$REPO.git"
  git push -q -f origin gh-pages )
rm -rf "$TMP"

echo "rolled out $REPO"
