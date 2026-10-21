#!/usr/bin/env bash
# EXPERIMENTAL, UNOFFICIAL replace mode (see README, "Experimental: replace mode"): the reference
# character is generated inside the driving video's own scene, then pasted onto the original background.
#
#   CKPT_DIR=<ckpts> DRIVE=<video> REF_IMG=<image> PROMPT_FILE=<prompt.txt> bash scripts/run_v100_replace.sh
#
# Required: CKPT_DIR, DRIVE, REF_IMG, and PROMPT (text) or PROMPT_FILE. Describe the new character and
#           the REAL scene of the driving video in the prompt's background part.
# Optional: GPUS=0,1,2,3  RES=480p  MODE=distill|base  SEED=42  OUT_DIR=~/ray_runs/replace
#           MASK_MODE=dilate|bbox (bbox: the hole is the person's box, no silhouette constraint)
#           DILATE=15 (px)  MASK=<.npz or mask video, white = target; made once if missing>
#           SEGMENTER=gsam2 (default: Grounded-SAM2, text TARGET -> box -> SAM2 masks tracked over the
#                      video; models GDINO / SAM2 = HF id or dir, default $CKPT_DIR/grounding-dino-base and
#                      $CKPT_DIR/sam2.1-hiera-large if present) | segformer (person only; MASK_MODEL)
#           TARGET=person   what to replace, in words (gsam2)
#           SCENE_REF=1 REF_TARGET="cat"  put the reference character on the video's own background
#                      first (no foreign backdrop such as a white studio wall gets copied)
#           SCALE=0.6  retarget: the driving person is rescaled around the feet so the character is
#                      drawn at that size (needs scripts/retarget_driving.py)
#           WORK_DIR=inputs  where masks and retargeted videos are written
# Output: <OUT_DIR>/<session>/results.mp4 (pasted onto the original) and results_raw.mp4 (model output)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
command -v python >/dev/null || { echo "python not found: activate the conda env first" >&2; exit 1; }
: "${CKPT_DIR:?set CKPT_DIR=<checkpoint folder>}"
: "${DRIVE:?set DRIVE=<driving video>}"
: "${REF_IMG:?set REF_IMG=<reference image>}"
if [[ -z ${PROMPT:-} ]]; then
  : "${PROMPT_FILE:?set PROMPT (text) or PROMPT_FILE}"
  PROMPT=$(cat "$PROMPT_FILE")
fi
DRIVE=$(realpath "$DRIVE"); REF_IMG=$(realpath "$REF_IMG")
GPUS=${GPUS:-0,1,2,3}
WORK_DIR=${WORK_DIR:-$REPO/inputs}
mkdir -p "$WORK_DIR"
SEGMENTER=${SEGMENTER:-gsam2}
TARGET=${TARGET:-person}
GDINO=${GDINO:-$CKPT_DIR/grounding-dino-base}; [[ -d $GDINO ]] || GDINO=IDEA-Research/grounding-dino-base
SAM2=${SAM2:-$CKPT_DIR/sam2.1-hiera-large}; [[ -d $SAM2 ]] || SAM2=facebook/sam2.1-hiera-large
GPU0=${GPUS%%,*}
if [[ $SEGMENTER == gsam2 ]]; then
  # own name per segmenter + target, so a SegFormer mask of the same video is never reused by mistake
  MASK=${MASK:-$WORK_DIR/$(basename "${DRIVE%.*}")_gsam2_$(echo "$TARGET" | tr -c 'A-Za-z0-9\n' '_')_mask.npz}
  [[ -f $MASK ]] || CUDA_VISIBLE_DEVICES=$GPU0 python scripts/grounded_sam2.py video --video "$DRIVE" \
    --target "$TARGET" --out "$MASK" --gdino "$GDINO" --sam2 "$SAM2"
elif [[ $SEGMENTER == segformer ]]; then
  MASK=${MASK:-$WORK_DIR/$(basename "${DRIVE%.*}")_person_mask.npz}
  MASK_MODEL=${MASK_MODEL:-$CKPT_DIR/segformer_b2_clothes}
  [[ -d $MASK_MODEL ]] || MASK_MODEL=mattmdjaga/segformer_b2_clothes
  [[ -f $MASK ]] || CUDA_VISIBLE_DEVICES=$GPU0 python scripts/make_person_mask.py --video "$DRIVE" \
    --out "$MASK" --model "$MASK_MODEL"
else
  echo "SEGMENTER must be gsam2 or segformer" >&2; exit 2
fi
if [[ ${SCENE_REF:-0} == 1 ]]; then  # the reference character on the video's own background
  SCENE=$WORK_DIR/$(basename "${REF_IMG%.*}")_scene.png
  CUDA_VISIBLE_DEVICES=$GPU0 python scripts/grounded_sam2.py scene-ref --ref "$REF_IMG" \
    --ref-target "${REF_TARGET:-character}" --video "$DRIVE" --masks "$MASK" --out "$SCENE" \
    --gdino "$GDINO" --sam2 "$SAM2"
  REF_IMG=$SCENE
fi

OUT_DIR=${OUT_DIR:-$HOME/ray_runs/replace}
BG_OVERRIDE=
if [[ -n ${SCALE:-} && ${SCALE} != 1 ]]; then  # retarget: motion from a rescaled copy, background from the original
  SCALE=$(python -c "import sys; print(f'{float(sys.argv[1]):g}')" "$SCALE")
  PREFIX=$WORK_DIR/$(basename "${DRIVE%.*}")
  python scripts/retarget_driving.py --video "$DRIVE" --masks "$MASK" --scale "$SCALE" --out-prefix "$PREFIX"
  BG_OVERRIDE="test_cfg.replace_background_video=$DRIVE"
  DRIVE=${PREFIX}_retarget_s${SCALE}.mp4
  MASK=${PREFIX}_hole_s${SCALE}.npz
  OUT_DIR=${OUT_DIR}_s${SCALE}
fi

export GPUS CKPT_DIR DRIVE REF_IMG OUT_DIR PROMPT
export MODE=${MODE:-distill} RES=${RES:-480p} SEED=${SEED:-42} MASTER_PORT=${MASTER_PORT:-29517}
export EXTRA_OVERRIDES="type=pipelines.wan_animate_2_replace_pipeline.WanAnimate2ReplacePipeline \
test_cfg.replace_mask=$MASK test_cfg.replace_dilate=${DILATE:-15} test_cfg.replace_mask_mode=${MASK_MODE:-dilate} \
test_cfg.dit_shard_dtype=bf16 test_cfg.log_gpu_memory=true $BG_OVERRIDE ${EXTRA_OVERRIDES:-}"
exec bash scripts/run_v100.sh
