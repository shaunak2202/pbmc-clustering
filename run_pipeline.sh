#!/usr/bin/env bash
# Run the complete PBMC clustering pipeline end-to-end.
# Usage:  bash run_pipeline.sh
# From repo root: /Users/shaunak/502/
#
# Step order rationale:
#   SC3 reads the preprocessed .h5ad checkpoint that each Python script
#   writes on its first run.  So each Python script must run BEFORE the
#   corresponding SC3 R script.  Python scripts are then re-run a second
#   time to pick up the SC3 labels for comparison plots.
#
#   1. pbmc_3k.py       — preprocess + all clustering (SC3 section skipped)
#   2. sc3_3k.R         — reads pbmc3k_preprocessed.h5ad → sc3_labels.csv
#   3. pbmc_3k.py       — re-run: loads checkpoint, adds SC3 comparison
#   4. pbmc_3k_magic.py — MAGIC pipeline (picks up sc3_labels.csv)
#   5. pbmc_10k.py      — preprocess + all clustering (SC3 section skipped)
#   6. sc3_10k.R        — reads pbmc10k_preprocessed.h5ad → sc3_labels_10k.csv
#   7. pbmc_10k.py      — re-run: loads checkpoint, adds SC3 comparison

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO_DIR/src"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

step() {
    echo -e "\n${CYAN}━━━ Step $1 / 7: $2 ${NC}"
}

ok() {
    echo -e "${GREEN}✓ Done in ${1}s${NC}"
}

fail() {
    echo -e "${RED}✗ Failed. See $LOG_DIR/$1.log${NC}"
    exit 1
}

cd "$REPO_DIR"

# ── Step 1: PBMC 3k baseline pipeline (first pass) ───────────
step 1 "PBMC 3k baseline pipeline — preprocessing + clustering"
T=$SECONDS
python "$SRC_DIR/pbmc_3k.py" 2>&1 | tee "$LOG_DIR/pbmc_3k_pass1.log" || fail pbmc_3k_pass1
ok $((SECONDS - T))

# ── Step 2: SC3 for PBMC 3k ──────────────────────────────────
step 2 "SC3 clustering — PBMC 3k  (reads checkpoint → sc3_labels.csv)"
T=$SECONDS
Rscript "$SRC_DIR/sc3_3k.R" 2>&1 | tee "$LOG_DIR/sc3_3k.log" || fail sc3_3k
ok $((SECONDS - T))

# ── Step 3: PBMC 3k baseline pipeline (second pass) ──────────
step 3 "PBMC 3k — re-run with SC3 labels (loads checkpoint, fast)"
T=$SECONDS
python "$SRC_DIR/pbmc_3k.py" 2>&1 | tee "$LOG_DIR/pbmc_3k_pass2.log" || fail pbmc_3k_pass2
ok $((SECONDS - T))

# ── Step 4: PBMC 3k + MAGIC pipeline ─────────────────────────
step 4 "PBMC 3k + MAGIC imputation pipeline"
T=$SECONDS
python "$SRC_DIR/pbmc_3k_magic.py" 2>&1 | tee "$LOG_DIR/pbmc_3k_magic.log" || fail pbmc_3k_magic
ok $((SECONDS - T))

# ── Step 5: PBMC 10k pipeline (first pass) ───────────────────
step 5 "PBMC 10k pipeline — preprocessing + clustering"
T=$SECONDS
python "$SRC_DIR/pbmc_10k.py" 2>&1 | tee "$LOG_DIR/pbmc_10k_pass1.log" || fail pbmc_10k_pass1
ok $((SECONDS - T))

# ── Step 6: SC3 for PBMC 10k ─────────────────────────────────
step 6 "SC3 clustering — PBMC 10k  (reads checkpoint → sc3_labels_10k.csv)"
T=$SECONDS
Rscript "$SRC_DIR/sc3_10k.R" 2>&1 | tee "$LOG_DIR/sc3_10k.log" || fail sc3_10k
ok $((SECONDS - T))

# ── Step 7: PBMC 10k pipeline (second pass) ──────────────────
step 7 "PBMC 10k — re-run with SC3 labels (loads checkpoint, fast)"
T=$SECONDS
python "$SRC_DIR/pbmc_10k.py" 2>&1 | tee "$LOG_DIR/pbmc_10k_pass2.log" || fail pbmc_10k_pass2
ok $((SECONDS - T))

echo -e "\n${GREEN}━━━ All steps complete ━━━${NC}"
echo "Figures : $SRC_DIR/figures/"
echo "Results : $SRC_DIR/results/"
echo "Logs    : $LOG_DIR/"
