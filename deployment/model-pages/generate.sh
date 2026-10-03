#!/usr/bin/env bash
# Generate one ready-to-apply patch per sibling repository from the catalog.
#
# Unlike deployment/rollout_all.sh (which pushes), this writes
# deployment/model-pages/<repo>.patch - a git-am-able commit that carries the
# rendered README.md, index.html and docs/index.html. It can be applied from any
# machine with write access to rajaram-2005/wt-pm-*, or attached to a PR.
#
# Usage:
#   bash deployment/model-pages/generate.sh [--check]
#
#   --check   only verify that the existing patches still apply to origin/main
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-$ROOT/build/patchwork}"
PAGES="${PAGES:-$ROOT/build/pages}"
OWNER="rajaram-2005"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

cd "$ROOT"
mkdir -p "$HERE" "$WORK"

if [ "$CHECK_ONLY" = "0" ]; then
  python3 deployment/build_model_pages.py --out "$PAGES" || exit 1
fi

mapfile -t REPOS < <(python3 - <<'PY'
import json
cat = json.load(open('deployment/models.catalog.json'))
for m in cat['models']:
    if m['repo'] != cat['reference_repo']:
        print(m['repo'])
PY
)

OK=0; BAD=0
for repo in "${REPOS[@]}"; do
  clone="$WORK/$repo"
  if [ ! -d "$clone/.git" ]; then
    git clone -q "https://github.com/$OWNER/$repo.git" "$clone" || { echo "FAIL clone $repo"; BAD=$((BAD+1)); continue; }
  fi
  (
    cd "$clone"
    git checkout -q main && git pull -q --ff-only origin main
    git checkout -q -- . 2>/dev/null || true
    patch="$HERE/$repo.patch"

    if [ "$CHECK_ONLY" = "0" ]; then
      git branch -D tmp-advanced-pages >/dev/null 2>&1 || true
      git checkout -q -b tmp-advanced-pages
      mkdir -p docs
      cp "$PAGES/$repo/README.md" README.md
      cp "$PAGES/$repo/index.html" index.html
      cp "$PAGES/$repo/docs/index.html" docs/index.html
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
      git format-patch -q -1 --stdout > "$patch"
      git checkout -q main
      git branch -D tmp-advanced-pages >/dev/null 2>&1 || true
    fi

    # The patch must apply to a clean main.
    if git apply --check "$patch" 2>/dev/null; then
      echo "OK   $repo  ($(wc -c < "$patch") bytes)"
      exit 0
    else
      echo "FAIL $repo  does not apply to origin/main"
      exit 1
    fi
  ) && OK=$((OK+1)) || BAD=$((BAD+1))
done

echo
echo "$OK patches ready in deployment/model-pages/, $BAD problem(s)"
[ "$BAD" -eq 0 ] || exit 1
