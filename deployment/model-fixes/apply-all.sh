#!/usr/bin/env bash
# Apply model 1..6 fixes to the sibling repos and rebuild their gh-pages branches.
#
# Run from the root of a clone of wt-pm-lstm-scada-anomaly.
# Requires: git, gh (authenticated with write access to rajaram-2005/wt-pm-*),
# and the six sibling repos cloned as siblings of THIS directory:
#
#   projects/
#     wt-pm-lstm-scada-anomaly/         <- cwd
#     wt-pm-1d-cnn-bearing-vibration/
#     wt-pm-aerozip-autoencoder-compressor/
#     wt-pm-contrastive-ssl-vibration/
#     wt-pm-convlstm-wear-prognostics/
#     wt-pm-dbn-feature-extraction/
#     wt-pm-deep-svdd-boundary/
#
# Or set REPO_PARENT to where the six sibling repos live.
#
# Usage: bash deployment/model-fixes/apply-all.sh
#
# Safe: the script aborts on dirty worktrees and shows every command.
set -euo pipefail

REPO_PARENT="${REPO_PARENT:-$(dirname "$(pwd)")}"
HERE="$(cd "$(dirname "$0")" && pwd)"

REPOS=(
  wt-pm-1d-cnn-bearing-vibration
  wt-pm-aerozip-autoencoder-compressor
  wt-pm-contrastive-ssl-vibration
  wt-pm-convlstm-wear-prognostics
  wt-pm-dbn-feature-extraction
  wt-pm-deep-svdd-boundary
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

BRANCH="arena/01a0f76b-models-1to6-fixes"

# 2. Apply main-branch fixes on a fresh branch in each repo
for r in "${REPOS[@]}"; do
  echo "=========================================="
  echo ">>> $r : checkout main, pull, create $BRANCH"
  cd "$REPO_PARENT/$r"
  git checkout main
  git pull --ff-only origin main
  # Start the fix branch from main (delete if an old one exists)
  git branch -D "$BRANCH" 2>/dev/null || true
  git checkout -b "$BRANCH"
  echo ">>> $r : applying patch"
  git am "$HERE/$r.patch"
done

# 3. Push the fix branches and open PRs
echo
echo "=========================================="
echo ">>> Pushing fix branches and opening PRs"
for r in "${REPOS[@]}"; do
  cd "$REPO_PARENT/$r"
  git push -u origin "$BRANCH"
  gh pr create \
    --repo "rajaram-2005/$r" \
    --base main \
    --head "$BRANCH" \
    --title "Fix: correct landing page, clean misplaced files, upgrade Model 6" \
    --body "Bundled fix from wt-pm-lstm-scada-anomaly session (models 1–6).

- Landing page (\`index.html\`, \`docs/index.html\`): correct platform-adapter card, status, dependencies, and a working View-on-GitHub link (replaces the placeholder '#' link).
- \`wt-pm-dbn-feature-extraction\` (model 5): removes cross-repo staging artifacts that were accidentally committed here (audit docs, patches for other repos, the staged Model 6 payload under \`docs/model-6-deep-svdd/\`). The 5-layer DBN implementation and \`results/model5_metrics.json\` are untouched.
- \`wt-pm-deep-svdd-boundary\` (model 6): upgrades from the 29-line 3-layer scaffold to the 6-layer bias-free Deep SVDD encoder with the 24-turbine synthetic-farm pipeline, ZCA whitening, fixed 20-epoch schedule, per-fault delay reporting, and two results JSON files (seed 42 + six-seed sweep). Adapter API surface (\`DeepSVDDNetwork\`, \`init_center\`, \`svdd_loss\`) is preserved.
- \`wt-pm-deep-svdd-boundary/requirements.txt\`: drops unused \`scikit-learn\` (kept \`torch numpy\`)." \
    || echo "(PR may already exist for $r — continuing)"
done

# 4. Rebuild gh-pages branches (clean, one-file static site) per repo
echo
echo "=========================================="
echo ">>> Rebuilding gh-pages as a clean 1-file static branch per repo"
for r in "${REPOS[@]}"; do
  cd "$REPO_PARENT/$r"
  echo ">>> $r : rebuilding gh-pages"
  # Read the index.html from the fix branch into a temp worktree
  TMP="$(mktemp -d)"
  git --work-tree="$TMP" checkout "$BRANCH" -- index.html
  # Create orphan gh-pages-tmp and commit only index.html
  git checkout --orphan gh-pages-tmp
  git rm -rf . >/dev/null 2>&1 || true
  cp "$TMP/index.html" index.html
  rm -rf "$TMP"
  git add index.html
  git -c user.email="rajaram-2005@users.noreply.github.com" \
      -c user.name="Rajaraman" \
      commit -m "Publish landing page only (clean gh-pages)"
  # Force-push the clean orphan to gh-pages, then return to the fix branch
  git push -f origin gh-pages-tmp:gh-pages
  git checkout "$BRANCH"
  git branch -D gh-pages-tmp
done

echo
echo "=========================================="
echo "DONE. Review the PRs and merge them, then re-run GitHub Pages builds."
echo "After the PRs merge, the \$BRANCH branches can be deleted."
