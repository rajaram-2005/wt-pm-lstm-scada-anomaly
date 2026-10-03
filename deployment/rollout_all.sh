#!/usr/bin/env bash
# Roll the 25-page "advanced concepts" content out to every sibling repository.
#
# One catalog (deployment/models.catalog.json) renders 25 pages. This script
# pushes those pages to the 24 sibling wt-pm-* repositories:
#
#   clone -> branch -> README + index.html + docs/index.html -> PR -> merge
#         -> rebuild gh-pages as a clean single-file site -> verify
#
# The reference repository (wt-pm-lstm-scada-anomaly, model 13) publishes its
# copy of all 25 pages through its own Pages workflow instead.
#
# Requires: git, gh, and a token with WRITE access to rajaram-2005/wt-pm-*.
# A read-only token fails at the push step with HTTP 403 - that is the
# environment limit, not a bug in this script.
#
# Usage:
#   bash deployment/rollout_all.sh --dry-run          # show the plan
#   bash deployment/rollout_all.sh                    # do it
#   bash deployment/rollout_all.sh --repo wt-pm-gru-scada-telemetry
#   bash deployment/rollout_all.sh --work /tmp/roll --pages /tmp/pages
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$ROOT/build/rollout}"
PAGES="${PAGES:-$ROOT/build/pages}"
OWNER="rajaram-2005"
REF_REPO="wt-pm-lstm-scada-anomaly"
BRANCH="arena/01a1001d-advanced-pages"
DRY=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --repo) ONLY="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --pages) PAGES="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

cd "$ROOT"
mkdir -p "$WORK"

# 1. Render every page from the catalog (idempotent, deterministic).
echo ">>> rendering pages from deployment/models.catalog.json"
python3 deployment/build_model_pages.py --out "$PAGES" || exit 1

# 2. Which repositories?
mapfile -t REPOS < <(python3 - <<'PY'
import json
cat = json.load(open('deployment/models.catalog.json'))
ref = cat['reference_repo']
for m in cat['models']:
    if m['repo'] != ref:
        print(m['repo'])
PY
)

if [ -n "$ONLY" ]; then
  REPOS=("$ONLY")
fi
echo ">>> ${#REPOS[@]} repositories; work dir $WORK"
if [ "$DRY" = "1" ]; then
  printf '    %s\n' "${REPOS[@]}"
  echo ">>> dry run: nothing pushed"
  exit 0
fi

# 3. Push one repository at a time, collect the outcome, never abort the batch.
FAILED=()
for repo in "${REPOS[@]}"; do
  clone="$WORK/$repo"
  if [ ! -d "$clone/.git" ]; then
    echo ">>> cloning $repo"
    git clone -q "https://github.com/$OWNER/$repo.git" "$clone" || { FAILED+=("$repo:clone"); continue; }
  fi
  echo "=========================================================="
  echo ">>> $repo"
  if bash deployment/rollout_one.sh "$repo" "$clone" "$PAGES"; then
    echo "    OK"
  else
    echo "    FAILED (see above)"
    FAILED+=("$repo")
  fi
done

# 4. Verify whatever landed.
echo
echo ">>> verifying against GitHub"
python3 deployment/check_ecosystem.py --remote || true

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "DONE: all ${#REPOS[@]} sibling repositories merged and published."
else
  echo "INCOMPLETE: ${#FAILED[@]} of ${#REPOS[@]} repositories failed:"
  printf '  - %s\n' "${FAILED[@]}"
  echo "If the failures are HTTP 403 on push, the GitHub token needs write"
  echo "access to $OWNER/wt-pm-*; reconnect GitHub in the session and re-run."
  exit 1
fi
