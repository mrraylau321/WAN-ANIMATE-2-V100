#!/usr/bin/env bash
# TI2V-5B in-context pilot, end to end on one node (default: 4x V100 32GB). Each stage skips work already done,
# so the same command resumes after a crash.
#   bash train/run_pilot_ti2v5b.sh                 filter -> select -> prep -> train -> infer
#   bash train/run_pilot_ti2v5b.sh train infer     only these stages
# Stages:
#   filter  Grounding DINO person boxes + motion on every clip (one shard per GPU)   -> $WORK/scores_<i>.csv
#   select  one main person, >= 40% of the frame height, drop the stillest 35%       -> $WORK/pass.txt
#   prep    text.pt once, then latents of 2 windows per passing clip (one shard per GPU) -> $WORK/cache
#   train   LoRA on the self-attention Q/K/V, torchrun over the GPUs                 -> $WORK/run/lora_*.safetensors
#   infer   official demo (reference cat + dancing template): trained LoRA vs untrained base -> $WORK/infer
# Knobs (env): VIDEOS CKPT_5B WORK GPUS STEPS FRAMES REF DRIVE GDINO T5_WAN T5_TOKENIZER MASTER_PORT
#   WEIGHTS=fp16  frozen weights stored in fp16 (10 GB instead of 20 GB per GPU) if train/infer runs out of memory
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/ma-user/anaconda3/envs/wan_animate_2}
[[ -d $ENV_PREFIX/bin ]] && export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
VIDEOS=${VIDEOS:-/cache/humanvid_real/vertical_static}
CKPT_5B=${CKPT_5B:-/cache/WAN2.2-TI2V-5B-Diffusers/WAN2.2-TI2V-5B-Diffusers}
WORK=${WORK:-/cache/pilot_ti2v5b}
GDINO=${GDINO:-$REPO/ckpts/grounding-dino-base}
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-2000}
FRAMES=${FRAMES:-33}
REF=${REF:-$REPO/examples/demo1/reference.png}
DRIVE=${DRIVE:-$REPO/examples/demo1/template.mp4}
STAGES=${*:-filter select prep train infer}
IFS=, read -r -a GPU <<< "$GPUS"
N=${#GPU[@]}
mkdir -p "$WORK"
cd "$REPO"
echo "repo $REPO | videos $VIDEOS | 5B $CKPT_5B | work $WORK | GPUs $GPUS | stages: $STAGES"
[[ -f $CKPT_5B/transformer/config.json && -f $CKPT_5B/vae/config.json ]] || { echo "TI2V-5B diffusers weights missing in $CKPT_5B" >&2; exit 1; }

shards() {  # run "$@ --shard i/N" on every GPU in parallel, fail if any shard fails
  local pids=() i
  for i in "${!GPU[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU[$i]} "$@" --shard "$i/$N" > "$WORK/$(basename "$2" .py)_$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "shard $i failed: see $WORK/$(basename "$2" .py)_$i.log" >&2; exit 1; }
  done
}

for stage in $STAGES; do
  case $stage in
    filter)
      [[ -d $VIDEOS ]] || { echo "no clips in $VIDEOS (set VIDEOS=<folder of mp4>)" >&2; exit 1; }
      [[ -d $GDINO ]] || echo "note: $GDINO not found, using the HF id (needs internet)" >&2
      [[ -d $GDINO ]] || GDINO=IDEA-Research/grounding-dino-base
      echo "== filter: $(ls "$VIDEOS"/*.mp4 | wc -l) clips on $N GPUs (logs $WORK/filter_clips_<i>.log)"
      pids=()
      for i in "${!GPU[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} python train/filter_clips.py score --videos "$VIDEOS" --gdino "$GDINO" \
          --out "$WORK/scores_$i.csv" --shard "$i/$N" > "$WORK/filter_clips_$i.log" 2>&1 &
        pids+=($!)
      done
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "filter shard $i failed: $WORK/filter_clips_$i.log" >&2; exit 1; }; done ;;
    select)
      python train/filter_clips.py select --scores "$WORK"/scores_*.csv --out "$WORK/pass.txt" ;;
    prep)
      [[ -f $WORK/pass.txt ]] || { echo "run filter + select first ($WORK/pass.txt missing)" >&2; exit 1; }
      T5=()
      [[ -n ${T5_WAN:-} ]] && T5=(--t5-wan "$T5_WAN" --t5-tokenizer "${T5_TOKENIZER:?set T5_TOKENIZER with T5_WAN}")
      if [[ ! -f $WORK/cache/text.pt ]]; then
        echo "== prep: text embedding"
        CUDA_VISIBLE_DEVICES=${GPU[0]} python train/prep_incontext_ti2v5b.py --model "$CKPT_5B" --videos "$VIDEOS" \
          --out "$WORK/cache" --text-only ${T5[@]+"${T5[@]}"}
      fi
      echo "== prep: $(grep -c . "$WORK/pass.txt") clips on $N GPUs (logs $WORK/prep_incontext_ti2v5b_<i>.log)"
      shards python train/prep_incontext_ti2v5b.py --model "$CKPT_5B" --videos "$VIDEOS" --list "$WORK/pass.txt" \
        --out "$WORK/cache" --frames "$FRAMES"
      echo "$(ls "$WORK/cache" | grep -c '_w[0-9]*\.pt$') samples in $WORK/cache" ;;
    train)
      last=$WORK/run/lora_$(printf %06d "$STEPS").safetensors
      if [[ -f $last ]]; then echo "== train: $last exists, skipped"; continue; fi
      echo "== train: $STEPS steps on $N GPUs (log $WORK/train.log)"
      CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node "$N" --master_port "${MASTER_PORT:-29531}" \
        train/train_incontext_ti2v5b.py --model "$CKPT_5B" --data "$WORK/cache" --out "$WORK/run" --steps "$STEPS" ${WEIGHTS:+--weights "$WEIGHTS"} \
        2>&1 | tee -a "$WORK/train.log" ;;
    infer)
      lora=$(ls "$WORK"/run/lora_*.safetensors 2>/dev/null | tail -1)
      [[ -n $lora ]] || { echo "no LoRA in $WORK/run: run train first" >&2; exit 1; }
      for l in "$lora" ""; do
        echo "== infer: ${l:-untrained base}"
        CUDA_VISIBLE_DEVICES=${GPU[0]} python train/infer_incontext_ti2v5b.py --model "$CKPT_5B" ${l:+--lora "$l"} \
          --text "$WORK/cache/text.pt" --ref "$REF" --drive "$DRIVE" --frames "$FRAMES" --out "$WORK/infer" ${WEIGHTS:+--weights "$WEIGHTS"}
      done ;;
    *) echo "unknown stage $stage (filter select prep train infer)" >&2; exit 2 ;;
  esac
done
echo "done: $STAGES"
