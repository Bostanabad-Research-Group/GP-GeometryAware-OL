#!/bin/bash
# Restores <ROOT_FOLDER> placeholders. Run from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# local.yaml: was <ROOT_FOLDER>/reefgp
sed -i "s|$REPO_ROOT|<ROOT_FOLDER>/reefgp|g" "$REPO_ROOT/configs/machine/local.yaml"

# run.sh and run_baselines.sh: python calls had <ROOT_FOLDER>/reefgp/olgp/...,
# PYTHONPATH lines had bare <ROOT_FOLDER> — restore in two passes (specific first)
for f in \
    "$REPO_ROOT/scripts/linux/run.sh" \
    "$REPO_ROOT/scripts/linux/run_baselines.sh"; do
    sed -i "s|$REPO_ROOT/olgp|<ROOT_FOLDER>/reefgp/olgp|g" "$f"
    sed -i "s|$REPO_ROOT|<ROOT_FOLDER>|g" "$f"
done

# Transolver scripts: plain <ROOT_FOLDER>
for f in \
    "$REPO_ROOT/scripts/linux/transolver/run_eval.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_train.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_preprocess.sh" \
    "$REPO_ROOT/scripts/linux/transolver/run_mmd.sh"; do
    sed -i "s|$REPO_ROOT|<ROOT_FOLDER>|g" "$f"
done

echo "Done. Restored <ROOT_FOLDER> placeholders (REPO_ROOT was: $REPO_ROOT)"
