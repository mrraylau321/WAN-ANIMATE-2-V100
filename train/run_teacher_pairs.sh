#!/usr/bin/env bash
# P2 teacher pairs for the TI2V-5B in-context LoRA: Wan-Animate-2 makes "person B does person A's motion", so the LoRA
# is trained on pairs where the driving person and the target person differ. Each stage skips finished work.
#   bash train/run_teacher_pairs.sh                    jobs -> generate -> prep
#   N=50 bash train/run_teacher_pairs.sh               a small first batch (look at a few results before scaling)
# Stages:
#   jobs      N pairs from $WORK/pass.txt: A = a 33-frame window (24 fps, 480x832 mp4), B = a frame of another clip
#                                                                               -> $WORK/teacher/{drive,ref,jobs.jsonl}
#   generate  Wan-Animate-2 move mode on all GPUs, model loaded once (distill, 10 steps) -> $WORK/teacher/teacher/<id>/
#   prep      TI2V-5B latents: target = [B, teacher frames 1..], drive = A degraded as in self-reconstruction
#                                                                               -> $WORK/$TCACHE
# Then train on both caches:
#   RUN=run_p2 DRIVE_ROPE=offset EXTRA_DATA=$TCACHE MIX="0.6 0.4" bash train/run_pilot_ti2v5b.sh train infer
# Knobs (env): N WORK VIDEOS CKPT_DIR (Wan-Animate-2) CKPT_5B GPUS FRAMES TCACHE DRIVE_NORM (match the self-recon
#   cache) T5_WAN T5_TOKENIZER MASTER_PORT
# Box: 4x V100 32GB -> GPUS=0,1,2,3, distill, bf16 DiT shards (same settings as scripts/run_v100_replace.sh).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/ma-user/anaconda3/envs/wan_animate_2}
[[ -d $ENV_PREFIX/bin ]] && export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
VIDEOS=${VIDEOS:-/cache/humanvid_real/vertical_static}
WORK=${WORK:-/cache/pilot_ti2v5b}
CKPT_DIR=${CKPT_DIR:-/home/ma-user/work/l00996894/ckpts}
CKPT_5B=${CKPT_5B:-/cache/WAN2.2-TI2V-5B-Diffusers/WAN2.2-TI2V-5B-Diffusers}
GPUS=${GPUS:-0,1,2,3}
N=${N:-300}
FRAMES=${FRAMES:-33}
TCACHE=${TCACHE:-cache_teacher}
NORM=()
[[ ${DRIVE_NORM:-0} == 1 ]] && NORM=(--drive-norm)
T5=()
[[ -n ${T5_WAN:-} ]] && T5=(--t5-wan "$T5_WAN" --t5-tokenizer "${T5_TOKENIZER:?set T5_TOKENIZER with T5_WAN}")
STAGES=${*:-jobs generate prep}
TDIR=$WORK/teacher
IFS=, read -r -a GPU <<< "$GPUS"
NGPU=${#GPU[@]}
cd "$REPO"
echo "teacher pairs: N=$N | work $WORK | Wan-Animate-2 $CKPT_DIR | TI2V-5B $CKPT_5B | GPUs $GPUS | stages: $STAGES"

pids=()
cleanup() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in ${pids[@]+"${pids[@]}"}; do kill "$pid" 2>/dev/null || true; done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for stage in $STAGES; do
  case $stage in
    jobs)
      [[ -f $WORK/pass.txt ]] || { echo "$WORK/pass.txt missing: run the pilot's filter + select first" >&2; exit 1; }
      python train/make_teacher_jobs.py --videos "$VIDEOS" --list "$WORK/pass.txt" --out "$TDIR" --n "$N" \
        --frames "$FRAMES" ;;
    generate)
      [[ -f $TDIR/jobs.jsonl ]] || { echo "run the jobs stage first" >&2; exit 1; }
      WAN=$CKPT_DIR/videomodel/Wan-AI
      DIT=$CKPT_DIR/wan_animate_2/wan_animate_2_bf16_distillation.safetensors
      for f in "$DIT" "$WAN/models_t5_umt5-xxl-enc-bf16.pth" "$WAN/vae.pth" \
               "$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$WAN/umt5-xxl" "$WAN/xlm-roberta-large"; do
        [[ -e $f ]] || { echo "missing: $f (set CKPT_DIR)" >&2; exit 1; }
      done
      (( 40 % NGPU == 0 )) || { echo "the 40 attention heads must divide by the GPU count ($NGPU)" >&2; exit 2; }
      echo "== generate: $(grep -c . "$TDIR/jobs.jsonl") jobs (log $TDIR/generate.log)"
      ( cd "$REPO/infer" && CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
          WAN_SDPA_Q_CHUNK=${WAN_SDPA_Q_CHUNK:-4096} MASTER_PORT=${MASTER_PORT:-29517} \
          python wan_animate_2_batch.py --jobs "$TDIR/jobs.jsonl" --config ./wan_animate_2_distillation_v100.yaml \
            --dtype auto --width 480 --height 832 --fps 24 --clip_len "$FRAMES" --step 10 --sample_guide_scale 1.0 \
            --cfg-override "model_path=$DIT" "model.t5_checkpoint=$WAN/models_t5_umt5-xxl-enc-bf16.pth" \
              "model.t5_tokenizer=$WAN/umt5-xxl" "model.vae_checkpoint=$WAN/vae.pth" \
              "model.clip_checkpoint=$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
              "model.clip_tokenizer=$WAN/xlm-roberta-large" "test_cfg.sp_size=$NGPU" "test_cfg.sharding_size=$NGPU" \
              test_cfg.dit_shard_dtype=bf16 test_cfg.log_gpu_memory=true ) 2>&1 | tee -a "$TDIR/generate.log" ;;
    prep)
      [[ -f $TDIR/jobs.jsonl ]] || { echo "run the jobs stage first" >&2; exit 1; }
      echo "== prep: text embedding + $(ls "$TDIR"/teacher/*/results.mp4 2>/dev/null | wc -l) teacher videos -> $WORK/$TCACHE"
      CUDA_VISIBLE_DEVICES=${GPU[0]} python train/prep_incontext_ti2v5b.py --model "$CKPT_5B" --teacher "$TDIR/jobs.jsonl" \
        --out "$WORK/$TCACHE" --frames "$FRAMES" --text-only ${T5[@]+"${T5[@]}"} ${NORM[@]+"${NORM[@]}"}
      pids=()
      for i in "${!GPU[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} python train/prep_incontext_ti2v5b.py --model "$CKPT_5B" --teacher "$TDIR/jobs.jsonl" \
          --out "$WORK/$TCACHE" --frames "$FRAMES" --shard "$i/$NGPU" ${T5[@]+"${T5[@]}"} ${NORM[@]+"${NORM[@]}"} \
          > "$WORK/prep_teacher_$i.log" 2>&1 &
        pids+=($!)
      done
      for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "prep shard $i failed: $WORK/prep_teacher_$i.log" >&2; exit 1; }
        unset 'pids[i]'
      done
      echo "$(ls "$WORK/$TCACHE" | grep -c '_w[0-9]*\.pt$') teacher samples in $WORK/$TCACHE" ;;
    *) echo "unknown stage $stage (jobs generate prep)" >&2; exit 2 ;;
  esac
done
echo "done: $STAGES"
