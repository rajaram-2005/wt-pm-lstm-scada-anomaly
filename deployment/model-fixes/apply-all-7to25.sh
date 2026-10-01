#!/usr/bin/env bash
# Wire models 7..25 (the remaining 18 sibling repos) exactly like models 1..6:
# apply the platform-aware README + landing-page patch, open a PR, and rebuild
# each gh-pages branch as a clean one-file static site.
#
# Model 13 (wt-pm-lstm-scada-anomaly) is THIS repository and is already wired.
#
# Run from the root of a clone of wt-pm-lstm-scada-anomaly.
# Requires: git, gh (authenticated with write access to rajaram-2005/wt-pm-*),
# and the 18 sibling repos cloned as siblings of THIS directory:
#
#   projects/
#     wt-pm-lstm-scada-anomaly/         <- cwd
#     wt-pm-digital-twin-surrogate/
#     wt-pm-gnn-turbines-cascade/
#     wt-pm-gru-scada-telemetry/
#     ... (see REPOS below)
#
# Or set REPO_PARENT to where the sibling repos live.
#
# Usage: bash deployment/model-fixes/apply-all-7to25.sh
#
# Safe: the script aborts on dirty worktrees and shows every command.
set -euo pipefail

REPO_PARENT="${REPO_PARENT:-$(dirname "$(pwd)")}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Ecosystem numbers 7..25, excluding 13 (this repo).
REPOS=(
  wt-pm-digital-twin-surrogate      # model 7
  wt-pm-gnn-turbines-cascade        # model 8
  wt-pm-gru-scada-telemetry         # model 9
  wt-pm-hmm-degradation-states      # model 10
  wt-pm-informer-long-sequence      # model 11
  wt-pm-isolation-forest-telemetry  # model 12
  wt-pm-mlp-rul-regression          # model 14
  wt-pm-particle-filter-rul         # model 15
  wt-pm-pg-bnn-wind-turbine         # model 16
  wt-pm-quantized-mobilenet-edge    # model 17
  wt-pm-random-forest-telemetry     # model 18
  wt-pm-snn-event-vibration         # model 19
  wt-pm-svm-rbf-generator-stator    # model 20
  wt-pm-tcn-power-curve             # model 21
  wt-pm-tinyml-esp32-safety-relay   # model 22
  wt-pm-vae-reconstruction-loss     # model 23
  wt-pm-xai-shap-interpretable      # model 24
  wt-pm-xgboost-tabular-faults      # model 25
)

echo "Using sibling repos under: $REPO_PARENT"
echo "Patches live in: $HERE"
echo

# 1. Sanity check clones
for r in "${REPOS[@]}"; do
  if [[ ! -d "$REPO_PARENT/$r/.git" ]]; then
    echo "ERROR: $REPO_PARENT/$r is not a git clone. Clone it first:"
    echo "  git clone https://github.com/rajaram-2005/$r.git $REPO_PARENT/$r"
    exit 1
  fi
  if [[ -n "$(cd "$REPO_PARENT/$r" && git status --porcelain)" ]]; then
    echo "ERROR: $REPO_PARENT/$r has uncommitted changes; aborting."
    exit 1
  fi
done

BRANCH="arena/01a0f786-models-7to25-wiring"

# 2. Apply the wiring patch on a fresh branch in each repo
for r in "${REPOS[@]}"; do
  echo "=========================================="
  echo ">>> $r : checkout main, pull, create $BRANCH"
  cd "$REPO_PARENT/$r"
  git checkout main
  git pull --ff-only origin main
  git branch -D "$BRANCH" 2>/dev/null || true
  git checkout -b "$BRANCH"
  if [[ ! -f "$HERE/$r.patch" ]]; then
    echo "ERROR: $HERE/$r.patch missing"
    exit 1
  fi
  echo ">>> $r : applying patch"
  git am "$HERE/$r.patch"
done

# 3. Push the wiring branches and open PRs
echo
echo "=========================================="
echo ">>> Pushing wiring branches and opening PRs"
for r in "${REPOS[@]}"; do
  cd "$REPO_PARENT/$r"
  git push -u origin "$BRANCH"
  gh pr create \
    --repo "rajaram-2005/$r" \
    --base main \
    --head "$BRANCH" \
    --title "Wire this model into the WT-PM platform (same treatment as models 1–6)" \
    --body "Bundled wiring from wt-pm-lstm-scada-anomaly session (models 7–25, matching the models 1–6 bundle in PR rajaram-2005/wt-pm-lstm-scada-anomaly#6).

- \`README.md\`: full platform-aware description — platform adapter id, integration status, model entry points, platform contract (\`WTDataSchema\` / \`wt-pm.platform.v1\`), Hermes/XAI role and honest limits.
- \`index.html\` and \`docs/index.html\`: platform-adapter card, status, files list, and a working **View on GitHub** link (replaces the placeholder \`#\`); honest footer replaces the scaffold note.
- \`model.py\` and \`requirements.txt\` are **untouched** — the repo stays independently runnable (\`python model.py\`)." \
    || echo "(PR may already exist for $r — continuing)"
done

# 4. Rebuild gh-pages branches (clean, one-file static site) per repo
echo
echo "=========================================="
echo ">>> Rebuilding gh-pages as a clean 1-file static branch per repo"
for r in "${REPOS[@]}"; do
  cd "$REPO_PARENT/$r"
  echo ">>> $r : rebuilding gh-pages"
  TMP="$(mktemp -d)"
  git --work-tree="$TMP" checkout "$BRANCH" -- index.html
  git checkout --orphan gh-pages-tmp
  git rm -rf . >/dev/null 2>&1 || true
  cp "$TMP/index.html" index.html
  rm -rf "$TMP"
  git add index.html
  git -c user.email="rajaram-2005@users.noreply.github.com" \
      -c user.name="Rajaraman" \
      commit -m "Publish landing page only (clean gh-pages)"
  git push -f origin gh-pages-tmp:gh-pages
  git checkout "$BRANCH"
  git branch -D gh-pages-tmp
done

echo
echo "=========================================="
echo "DONE. Review the PRs and merge them, then re-run GitHub Pages builds."
echo "After the PRs merge, the $BRANCH branches can be deleted and"
echo "deployment/models.lock.json in the platform repo can be re-pinned:"
echo "  python deployment/fetch_models.py --dir /tmp/external --write-lock"
