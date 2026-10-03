#!/usr/bin/env bash
# Apply the ready-to-commit page patches to the 24 sibling repositories.
#
# Patches in this directory were generated from deployment/models.catalog.json by
# generate.sh. Each one replaces the repository's README.md, index.html and
# docs/index.html with the rendered, catalog-consistent page. model.py and
# requirements.txt are untouched.
#
#   clone -> git am -> push branch -> PR -> merge -> rebuild gh-pages -> verify
#
# Requires write access to rajaram-2005/wt-pm-*; a read-only token stops at the
# push step with HTTP 403.
#
# Usage:
#   bash deployment/model-pages/apply-all.sh --dry-run
#   bash deployment/model-pages/apply-all.sh
#   bash deployment/model-pages/apply-all.sh --repo wt-pm-gru-scada-telemetry
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-$ROOT/build/patchwork}"
PAGES="${PAGES:-$ROOT/build/pages}"
OWNER="rajaram-2005"
BRANCH="arena/01a1001d-advanced-pages"
DRY=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --repo) ONLY="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

cd "$ROOT"
mkdir -p "$WORK"

mapfile -t REPOS < <(python3 - <<'PY'
import json
cat = json.load(open('deployment/models.catalog.json'))
for m in cat['models']:
    if m['repo'] != cat['reference_repo']:
        print(m['repo'])
PY
)
[ -n "$ONLY" ] && REPOS=("$ONLY")

echo ">>> ${#REPOS[@]} repositories from $HERE"
if [ "$DRY" = "1" ]; then
  for repo in "${REPOS[@]}"; do
    printf '    %-42s %s\n' "$repo" "$([ -f "$HERE/$repo.patch" ] && echo "patch ready" || echo "PATCH MISSING")"
  done
  echo ">>> dry run: nothing pushed"
  exit 0
fi

# The gh-pages copy of a page is the same single file, taken straight from the render.
if [ ! -d "$PAGES" ]; then
  python3 deployment/build_model_pages.py --out "$PAGES" || exit 1
fi

FAILED=()
for repo in "${REPOS[@]}"; do
  patch="$HERE/$repo.patch"
  clone="$WORK/$repo"
  echo "=========================================================="
  echo ">>> $repo"
  if [ ! -f "$patch" ]; then echo "    FAILED: patch missing"; FAILED+=("$repo:patch"); continue; fi
  if [ ! -d "$clone/.git" ]; then
    git clone -q "https://github.com/$OWNER/$repo.git" "$clone" || { FAILED+=("$repo:clone"); continue; }
  fi

  if ! (
    set -e
    cd "$clone"
    git checkout -q main
    git pull -q --ff-only origin main
    git branch -D "$BRANCH" >/dev/null 2>&1 || true
    git checkout -q -b "$BRANCH"
    git am "$patch" 2>&1 | tail -2
    git push -q -u origin "$BRANCH"
  ); then
    echo "    FAILED at branch/push (HTTP 403 here means the token has no write access)"
    FAILED+=("$repo")
    continue
  fi

  gh pr create --repo "$OWNER/$repo" --base main --head "$BRANCH" \
    --title "Advanced-concepts page rendered from the collection catalog" \
    --body "Generated from \`deployment/models.catalog.json\` in
\`wt-pm-lstm-scada-anomaly\` (single source of truth for all 25 pages).

- Collection numbering stated both ways (alphabetical 01-25 and the task-based
  adapter id), so repository pages and the platform registry can be read against
  each other.
- Architecture table, 3-4 advanced concepts with failure modes, platform
  contract, platform-layer participation, cross-links to all 25 pages, honest
  limits and the fidelity rung.
- Self-contained landing page (\`index.html\`, \`docs/index.html\`): no external
  CSS/JS/assets, working View on GitHub link, replaces the scaffold page.
- \`model.py\` and \`requirements.txt\` untouched." \
    >/dev/null 2>&1 || echo "    (PR already exists)"

  gh pr merge "$BRANCH" --repo "$OWNER/$repo" --squash --delete-branch \
    >/dev/null 2>&1 || gh pr merge "$BRANCH" --repo "$OWNER/$repo" --squash --admin --delete-branch \
    || { echo "    FAILED at merge"; FAILED+=("$repo:merge"); continue; }

  # Clean single-file gh-pages.
  TMP="$(mktemp -d)"
  if [ -f "$PAGES/$repo/index.html" ]; then
    cp "$PAGES/$repo/index.html" "$TMP/index.html"
  else
    git -C "$clone" --work-tree="$TMP" checkout "$BRANCH" -- index.html
  fi
  (
    set -e
    cd "$TMP"
    git init -q -b gh-pages
    git add index.html
    git -c user.email="rajaram-2005@users.noreply.github.com" -c user.name="Rajaraman" \
        commit -q -m "Publish landing page only (clean gh-pages)"
    git remote add origin "https://github.com/$OWNER/$repo.git"
    git push -q -f origin gh-pages
  ) || { echo "    FAILED at gh-pages"; FAILED+=("$repo:gh-pages"); rm -rf "$TMP"; continue; }
  rm -rf "$TMP"

  cd "$ROOT"
  echo "    OK"
done

echo
echo ">>> verifying against GitHub"
python3 deployment/check_ecosystem.py --remote || true

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "DONE: all ${#REPOS[@]} sibling pages merged and published."
else
  echo "INCOMPLETE: ${#FAILED[@]} of ${#REPOS[@]} failed:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
