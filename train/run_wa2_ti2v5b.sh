#!/usr/bin/env bash
# Shared-QKV WA2 port. Reuse pass.txt from the pilot's filter/select stages.
#   WORK=/cache/pilot_ti2v5b bash train/run_wa2_ti2v5b.sh prep train infer
# prep   holds the last HOLDOUT (4) clips of pass.txt out of training (wa2_holdout.txt / wa2_train.txt), then caches
#        target = VAE(ref + clip[1:]) and drive = person-blurred VAE(clip) on every GPU -> $WORK/$CACHE
#        (+ PREVIEW=4 videos $WORK/$CACHE/preview_*.mp4: mask overlay | decoded blurred drive; look before training)
# train  torchrun on all GPUs -> $WORK/$RUN/qkv_<step>.safetensors
# infer  latest QKV on the most-moving held-out clip and the most-moving training clip (the pilot filter's motion score,
#        $WORK/scores_*.csv), reference = the clip's own frame 1 s after the window;
#        controls: drive | none (no driving) | reverse (time-reversed driving) | wrong (another clip's driving)
#        -> $WORK/infer_wa2/<held|train>_<clip>_<control>.mp4 (+ _side.mp4 = blurred drive | output), 4 runs at a time
# Knobs (env): VIDEOS CKPT_5B CKPT_DIR GDINO SAM2 WORK GPUS STEPS RUN CACHE WEIGHTS BLUR_SIGMA HOLDOUT PREVIEW REUSE_TARGET
#   SAVE_EVERY (500) SYNC_TO (obs://bucket/path: every save is also copied there with MoXing; a fresh machine resumes
#   from it)
#   T5_WAN T5_TOKENIZER MASTER_PORT
# Overfit test (does the training reach the driving branch at all?), after prep:
#   OVERFIT=8 bash train/run_wa2_ti2v5b.sh train infer
#   train  on the 8 samples of the most-moving training clips ($WORK/$RUN/overfit_samples.txt), RUN=run_overfit8,
#          STEPS=400, SAVE_EVERY=200 unless set
#   infer  on 2 of those samples straight from the cache (--sample: no VAE encode / masks), controls drive | none |
#          wrong (the other sample's drive) | base (untrained model, drive) -> $WORK/infer_wa2/overfit_*_side.mp4
#          = blurred drive | output | ground truth. Pass: drive ≈ ground truth, none / wrong / base clearly differ.
set -euo pipefail
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
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/ma-user/anaconda3/envs/wan_animate_2}
[[ -d $ENV_PREFIX/bin ]] && export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
VIDEOS=${VIDEOS:-/cache/humanvid_real/vertical_static}
CKPT_5B=${CKPT_5B:-/cache/WAN2.2-TI2V-5B-Diffusers/WAN2.2-TI2V-5B-Diffusers}
WORK=${WORK:-/cache/pilot_ti2v5b}
GPUS=${GPUS:-0,1,2,3}
OVERFIT=${OVERFIT:-}
if [[ -n $OVERFIT ]]; then
  (( OVERFIT >= 2 )) || { echo "OVERFIT needs at least 2 samples (one is the other's wrong drive)" >&2; exit 2; }
  RUN=${RUN:-run_overfit$OVERFIT}; STEPS=${STEPS:-400}; SAVE_EVERY=${SAVE_EVERY:-200}
fi
STEPS=${STEPS:-2000}
RUN=${RUN:-run_wa2}
CACHE=${CACHE:-cache_wa2}
WEIGHTS=${WEIGHTS:-fp16}
BLUR_SIGMA=${BLUR_SIGMA:-2.0}
FRAMES=${FRAMES:-33}
HOLDOUT=${HOLDOUT:-4}
PREVIEW=${PREVIEW:-4}
CKPT_DIR=${CKPT_DIR:-/home/ma-user/work/l00996894/ckpts}
mask_model() {  # name hf_id: $CKPT_DIR/<name>, else <repo>/ckpts/<name> (the pilot's filter), else the HF id (download)
  local d
  for d in "$CKPT_DIR/$1" "$REPO/ckpts/$1"; do [[ -d $d ]] && { echo "$d"; return; }; done
  echo "note: $1 not in $CKPT_DIR or $REPO/ckpts: downloading $2 from HF (needs internet; or set it)" >&2
  echo "$2"
}
GDINO=${GDINO:-$(mask_model grounding-dino-base IDEA-Research/grounding-dino-base)}
SAM2=${SAM2:-$(mask_model sam2.1-hiera-large facebook/sam2.1-hiera-large)}
STAGES=${*:-prep train infer}
IFS=, read -r -a GPU <<< "$GPUS"
N=${#GPU[@]}
mkdir -p "$WORK"
cd "$REPO"
[[ $FRAMES == 33 ]] || { echo "WA2 uses FRAMES=33, 480x832" >&2; exit 1; }
[[ -f $CKPT_5B/transformer/config.json && -f $CKPT_5B/vae/config.json ]] || { echo "local TI2V-5B weights missing in $CKPT_5B" >&2; exit 1; }
logs=()
wait_all() {  # every background job of this batch must succeed
  local i
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "failed: see ${logs[$i]}" >&2; exit 1; }
    unset 'pids[i]'
  done
  pids=(); logs=()
}
shards() {
  local i
  for ((i = 0; i < N; i++)); do
    logs+=("$WORK/prep_wa2_ti2v5b_$i.log")
    CUDA_VISIBLE_DEVICES=${GPU[$i]} "$@" --shard "$i/$N" > "${logs[$i]}" 2>&1 &
    pids+=($!)
  done
  wait_all
}
moving() {  # list n: the n clips of the list that move most (filter_clips.py motion score; list order without scores)
  python - "$1" "$2" "$WORK"/scores_*.csv <<'PY'
import csv, os, sys
names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
motion = {r["name"]: float(r["motion"]) for f in sys.argv[3:] if os.path.exists(f) for r in csv.DictReader(open(f))}
print("\n".join(sorted(names, key=lambda n: -motion.get(n, 0.0))[:int(sys.argv[2])]))
PY
}
runs=0
infer_sample() {  # sample other: the four overfit controls on one cached training sample, one GPU each
  local s=$1 other=$2 control name
  local -a extra q
  for control in drive none wrong base; do
    extra=(); q=(--qkv "$qkv")
    [[ $control == none ]] && extra=(--no-drive)
    [[ $control == wrong ]] && extra=(--drive-sample "$WORK/$CACHE/samples/$other")
    [[ $control == base ]] && q=()
    name=overfit_${s%.pt}_$control
    logs+=("$WORK/infer_wa2/$name.log")
    CUDA_VISIBLE_DEVICES=${GPU[$((runs % N))]} python train/infer_wa2_ti2v5b.py --model "$CKPT_5B" ${q[@]+"${q[@]}"} \
      --text "$WORK/$CACHE/text.pt" --sample "$WORK/$CACHE/samples/$s" --frames "$FRAMES" --out "$WORK/infer_wa2" \
      --name "$name" --weights "$WEIGHTS" --blur-sigma "$BLUR_SIGMA" ${extra[@]+"${extra[@]}"} \
      > "$WORK/infer_wa2/$name.log" 2>&1 &
    pids+=($!)
    runs=$((runs + 1))
    (( runs % N )) || wait_all
  done
}
infer_clip() {  # kind clip other: the four controls of one clip, one GPU each
  local kind=$1 clip=$2 other=$3 control name
  local -a extra
  for control in drive none reverse wrong; do
    extra=()
    [[ $control == none ]] && extra=(--no-drive)
    [[ $control == reverse ]] && extra=(--reverse-drive)
    [[ $control == wrong ]] && extra=(--drive-clip "$VIDEOS/$other")
    name=${kind}_${clip%.*}_$control
    logs+=("$WORK/infer_wa2/$name.log")
    CUDA_VISIBLE_DEVICES=${GPU[$((runs % N))]} python train/infer_wa2_ti2v5b.py --model "$CKPT_5B" --qkv "$qkv" \
      --text "$WORK/$CACHE/text.pt" --drive "$VIDEOS/$clip" --ref-from-drive --frames "$FRAMES" \
      --out "$WORK/infer_wa2" --name "$name" --weights "$WEIGHTS" --blur-sigma "$BLUR_SIGMA" --gdino "$GDINO" \
      --sam2 "$SAM2" ${extra[@]+"${extra[@]}"} > "$WORK/infer_wa2/$name.log" 2>&1 &
    pids+=($!)
    runs=$((runs + 1))
    (( runs % N )) || wait_all
  done
}
for stage in $STAGES; do
  case $stage in
    prep)
      [[ -f $WORK/pass.txt ]] || { echo "run pilot filter + select first ($WORK/pass.txt missing)" >&2; exit 1; }
      n=$(grep -c . "$WORK/pass.txt")
      (( HOLDOUT >= 2 && n > HOLDOUT )) || { echo "need HOLDOUT >= 2 and more than $HOLDOUT clips in pass.txt" >&2; exit 1; }
      split=$WORK/.wa2_split.$$
      grep . "$WORK/pass.txt" | awk -v m=$((n - HOLDOUT)) -v t="$split.train" -v h="$split.holdout" \
        '{ print > (NR <= m ? t : h) }'
      for part in train holdout; do  # an existing split is never changed: a model may already be trained on it
        if [[ -f $WORK/wa2_$part.txt ]] && ! cmp -s "$WORK/wa2_$part.txt" "$split.$part"; then
          rm -f "$split".*
          echo "$WORK/wa2_$part.txt has other clips (pass.txt or HOLDOUT changed): use a new WORK" >&2; exit 1
        fi
      done
      mv "$split.train" "$WORK/wa2_train.txt"; mv "$split.holdout" "$WORK/wa2_holdout.txt"
      T5=()
      [[ -n ${T5_WAN:-} ]] && T5=(--t5-wan "$T5_WAN" --t5-tokenizer "${T5_TOKENIZER:?set T5_TOKENIZER with T5_WAN}")
      REUSE=()
      if [[ -n ${REUSE_TARGET:-} ]]; then
        case $REUSE_TARGET in /*) REUSE=(--reuse-target "$REUSE_TARGET");;
          *) REUSE=(--reuse-target "$WORK/$REUSE_TARGET");; esac
      fi
      CUDA_VISIBLE_DEVICES=${GPU[0]} python train/prep_wa2_ti2v5b.py --model "$CKPT_5B" --videos "$VIDEOS" \
        --list "$WORK/wa2_train.txt" --out "$WORK/$CACHE" --frames "$FRAMES" --blur-sigma "$BLUR_SIGMA" \
        --gdino "$GDINO" --sam2 "$SAM2" --text-only ${T5[@]+"${T5[@]}"}
      echo "== prep: $(grep -c . "$WORK/wa2_train.txt") clips on $N GPUs -> $WORK/$CACHE/samples (progress: tail -f $WORK/prep_wa2_ti2v5b_0.log)"
      shards python train/prep_wa2_ti2v5b.py --model "$CKPT_5B" --videos "$VIDEOS" --list "$WORK/wa2_train.txt" \
        --out "$WORK/$CACHE" --frames "$FRAMES" --blur-sigma "$BLUR_SIGMA" --gdino "$GDINO" --sam2 "$SAM2" \
        --preview "$PREVIEW" ${T5[@]+"${T5[@]}"} ${REUSE[@]+"${REUSE[@]}"} ;;
    train)
      last=$WORK/$RUN/qkv_$(printf %06d "$STEPS").safetensors
      if [[ -f $last ]]; then echo "== train: $last exists, skipped"; continue; fi
      ONLY=()
      if [[ -n $OVERFIT ]]; then  # the samples of the most-moving training clips, both windows, fixed once per RUN
        list=$WORK/$RUN/overfit_samples.txt
        mkdir -p "$WORK/$RUN"
        if [[ ! -f $list ]]; then
          k=0; : > "$list.tmp"
          while read -r clip; do
            for w in 0 1; do
              f=${clip%.*}_w$w.pt
              if (( k < OVERFIT )) && [[ -f $WORK/$CACHE/samples/$f ]]; then echo "$f" >> "$list.tmp"; k=$((k + 1)); fi
            done
          done < <(moving "$WORK/wa2_train.txt" 1000000)
          (( k >= 2 )) || { rm -f "$list.tmp"; echo "fewer than 2 samples found in $WORK/$CACHE/samples" >&2; exit 1; }
          mv "$list.tmp" "$list"
        fi
        echo "== overfit: $(grep -c . "$list") samples ($list), $STEPS steps -> $WORK/$RUN"
        ONLY=(--only "$list")
      fi
      CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node "$N" --master_port "${MASTER_PORT:-29532}" \
        train/train_wa2_ti2v5b.py --model "$CKPT_5B" --data "$WORK/$CACHE" --out "$WORK/$RUN" \
        --steps "$STEPS" --weights "$WEIGHTS" --save-every "${SAVE_EVERY:-500}" ${SYNC_TO:+--sync-to "$SYNC_TO"} \
        ${ONLY[@]+"${ONLY[@]}"} 2>&1 | tee -a "$WORK/train_wa2.log" ;;
    infer)
      qkv=$(ls "$WORK/$RUN"/qkv_*.safetensors 2>/dev/null | tail -1 || true)
      [[ -n $qkv ]] || { echo "no QKV in $WORK/$RUN: run train first" >&2; exit 1; }
      mkdir -p "$WORK/infer_wa2"
      if [[ -n $OVERFIT ]]; then
        list=$WORK/$RUN/overfit_samples.txt
        [[ -f $list ]] || { echo "no $list: run OVERFIT=$OVERFIT ... train first" >&2; exit 1; }
        s1=$(sed -n 1p "$list"); s2=$(sed -n 2p "$list")
        echo "== overfit infer $qkv: $s1, $s2 -> $WORK/infer_wa2/overfit_*_side.mp4 (drive | output | ground truth)"
        infer_sample "$s1" "$s2"
        infer_sample "$s2" "$s1"
        wait_all
        continue
      fi
      [[ -f $WORK/wa2_holdout.txt && -f $WORK/wa2_train.txt ]] || { echo "run prep first (no holdout split)" >&2; exit 1; }
      held=$(moving "$WORK/wa2_holdout.txt" 2 | sed -n 1p); held2=$(moving "$WORK/wa2_holdout.txt" 2 | sed -n 2p)
      seen=$(moving "$WORK/wa2_train.txt" 1)
      mkdir -p "$WORK/infer_wa2"
      echo "== infer $qkv: held-out $held, training clip $seen -> $WORK/infer_wa2"
      infer_clip held "$held" "$held2"
      infer_clip train "$seen" "$held"
      wait_all ;;
    *) echo "unknown stage $stage (prep train infer)" >&2; exit 2 ;;
  esac
done
echo "done: $STAGES"
