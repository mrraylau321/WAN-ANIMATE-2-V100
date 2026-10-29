#!/usr/bin/env bash
# Wan-Animate-2 (14B) MOVE mode on a few named cases: the reference photo does the driving clip's motion in the
# photo's own scene ("live photo from one still": template motion + the user's photo). Plain run_v100.sh underneath.
#   bash scripts/run_move_cases.sh                       all cases below
#   bash scripts/run_move_cases.sh pink_girl__heart_hands one case
# Inputs: $IN (default <repo>/inputs/move_mode) with the renamed files
#   ref_beach_man.jpg  ref_pink_girl.jpg  drive_beach_jump.mp4  drive_beach_kick.mp4  drive_heart_hands.mp4
# Output: ~/ray_runs/move/<case>/<session>/results.mp4, plus a copy at ~/ray_runs/move/<case>.mp4
# Box: 4x V100 32GB -> GPUS=0,1,2,3, MODE=distill, bf16 DiT shards (same settings as scripts/run_v100_replace.sh).
# Knobs (env): IN CKPT_DIR GPUS MODE RES PROMPT (default " " = no prompt) SEED
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/ma-user/anaconda3/envs/wan_animate_2}
[[ -d $ENV_PREFIX/bin ]] && export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
IN=${IN:-$REPO/inputs/move_mode}
export CKPT_DIR=${CKPT_DIR:-/home/ma-user/work/l00996894/ckpts}
export GPUS=${GPUS:-0,1,2,3} MODE=${MODE:-distill} RES=${RES:-480p} MASTER_PORT=${MASTER_PORT:-29517}
export EXTRA_OVERRIDES="test_cfg.dit_shard_dtype=bf16 test_cfg.log_gpu_memory=true ${EXTRA_OVERRIDES:-}"
export PROMPT=${PROMPT:- }
OUT=${OUT:-$HOME/ray_runs/move}

# case name | reference photo | driving clip
CASES="beach_man__beach_jump|ref_beach_man.jpg|drive_beach_jump.mp4
beach_man__beach_kick|ref_beach_man.jpg|drive_beach_kick.mp4
pink_girl__heart_hands|ref_pink_girl.jpg|drive_heart_hands.mp4"

want=" $* "
ran=0
while IFS='|' read -r name ref drive; do
  [[ $# -gt 0 && $want != *" $name "* ]] && continue
  for f in "$IN/$ref" "$IN/$drive"; do [[ -f $f ]] || { echo "missing input: $f" >&2; exit 1; }; done
  echo "=== $name: $ref does the motion of $drive"
  OUT_DIR=$OUT/$name REF_IMG=$IN/$ref DRIVE=$IN/$drive bash "$REPO/scripts/run_v100.sh"
  latest=$(ls -td "$OUT/$name"/session_* 2>/dev/null | head -1)
  if [[ -f $latest/results.mp4 ]]; then cp "$latest/results.mp4" "$OUT/$name.mp4"; echo "-> $OUT/$name.mp4"; fi
  ran=$((ran + 1))
done <<< "$CASES"
(( ran > 0 )) || { echo "no case matched: $* (cases: $(cut -d'|' -f1 <<< "$CASES" | tr '\n' ' '))" >&2; exit 2; }
echo "done: $ran case(s) in $OUT"
