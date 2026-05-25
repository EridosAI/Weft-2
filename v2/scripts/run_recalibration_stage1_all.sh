#!/usr/bin/env bash
# V2 recalibration Stage 1 launcher — 7 cells x 3 seeds = 21 grok-curve runs.
# Sliding window of 2 concurrent processes (VRAM-probed safe: L_d=1 ~1.25GB,
# L_d=4 ~1.98GB; seed-major ordering spaces the heavier C3/L_d=4 seeds apart).
# Per-job logs in v2/logs/ (gitignored). Auto-aggregates lock_decision.json at end.
# Does NOT write the config V2_TRAINING_STEPS lock — that is a deliberate
# separate step (--write-lock) after reviewing lock_decision.json.

set -u
REPO="/mnt/c/Users/Jason/Desktop/Eridos/Weft 2"
cd "$REPO" || exit 1
MAXP=2
LOGDIR="v2/logs"
MASTER="$LOGDIR/recalib_stage1_master.log"
mkdir -p "$LOGDIR"
RUNNER="v2/scripts/run_recalibration_stage1.py"

log(){ echo "[launch $(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

# Seed-major ordering: C3 seeds land at well-spaced positions (3, 10, 17 of 21)
# so two L_d=4 runs rarely overlap.
JOBS=()
for S in 0 1 2; do for C in C1 C2 C3 C4 C5 C6 C7; do JOBS+=("$C $S"); done; done

run_job(){
  local C="$1" S="$2"
  log "START $C seed$S"
  python3 "$RUNNER" --cell "$C" --seed "$S" > "$LOGDIR/recalib_stage1_${C}_seed${S}.log" 2>&1
  log "DONE  $C seed$S rc=$?"
}

log "Stage 1 launch: ${#JOBS[@]} jobs, MAXP=$MAXP"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$MASTER"

for job in "${JOBS[@]}"; do
  read -r C S <<< "$job"
  run_job "$C" "$S" &
  while (( $(jobs -r | wc -l) >= MAXP )); do wait -n; done
done
wait

log "ALL ${#JOBS[@]} JOBS COMPLETE — aggregating"
python3 "$RUNNER" --aggregate >> "$MASTER" 2>&1
log "AGGREGATE COMPLETE — see v2/results/recalibration/stage1/lock_decision.json"
