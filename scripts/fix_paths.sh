#!/bin/bash
# Run from the repo root: bash scripts/fix_paths.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# local.yaml uses <ROOT_FOLDER>/reefgp — replace the whole thing with REPO_ROOT
sed -i "s|<ROOT_FOLDER>/reefgp|$REPO_ROOT|g" "$REPO_ROOT/configs/machine/local.yaml"

# run.sh mixes both styles: fix <ROOT_FOLDER>/reefgp/ first, then bare <ROOT_FOLDER>
sed -i "s|<ROOT_FOLDER>/reefgp|$REPO_ROOT|g" "$REPO_ROOT/scripts/linux/run.sh"
sed -i "s|<ROOT_FOLDER>|$REPO_ROOT|g"         "$REPO_ROOT/scripts/linux/run.sh"
sed -i "s|<ROOT_FOLDER>/reefgp|$REPO_ROOT|g" "$REPO_ROOT/scripts/linux/run_baselines.sh"
sed -i "s|<ROOT_FOLDER>|$REPO_ROOT|g"         "$REPO_ROOT/scripts/linux/run_baselines.sh"

# Transolver scripts use <ROOT_FOLDER> as repo root directly
for f in \
    "$REPO_ROOT/scripts/linux/transolver/run_eval.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_train.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_preprocess.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_mmd.sh"; do
    sed -i "s|<ROOT_FOLDER>|$REPO_ROOT|g" "$f"
done

echo "Done. REPO_ROOT set to: $REPO_ROOT"
