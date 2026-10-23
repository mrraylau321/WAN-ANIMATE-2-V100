#!/usr/bin/env bash
# Wan-Animate-2 inference on one node with 8x V100 32GB (sm70): fp16 DiT + SDPA attention,
# fp32 T5/VAE, fp16 CLIP (see README.md). Called by run_v100_base.sh / run_v100_distill.sh.
#
# Launch model: ONE python process per node; the pipeline itself mp.spawn()s one worker per visible
# GPU (upstream design). Do NOT use `torchrun --nproc_per_node=8` (that would create 8x8 ranks).
#
# Environment knobs (defaults in brackets):
#   MODE       base|distill [base]          RES   480p|540p|720p|<W>x<H> [480p] (portrait, like the example)
#   CKPT_DIR   [<repo>/ckpts]  layout of `huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir ckpts`
#   REF_IMG    [<repo>/examples/demo1/reference.png]   DRIVE [<repo>/examples/demo1/template.mp4]
#   PROMPT     caption of REF_IMG in the README's 人物外观描述/背景描述 format
#              [contents of <repo>/examples/demo1/prompt.txt]; set it whenever you change REF_IMG
#   STEPS / GUIDE  [base 40 / 3.0, distill 10 / 1.0]   CLIP_LEN [81]  FPS [24]  SEED [42]
#   DTYPE      auto|fp16|fp32|bf16 [auto = fp16 on V100]   GPUS [0,1,2,3,4,5,6,7]   OUT_DIR [<repo>/outputs]
#   EXTRA_OVERRIDES  extra OmegaConf dotlist, e.g. "test_cfg.overflow_action=abort test_cfg.fp32_modules=[blocks.17]"
set -euo pipefail

MODE=${MODE:-base}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEMO=$REPO/examples/demo1
CKPT_DIR=$(cd "${CKPT_DIR:-$REPO/ckpts}" && pwd)
REF_IMG=$(realpath "${REF_IMG:-$DEMO/reference.png}")
DRIVE=$(realpath "${DRIVE:-$DEMO/template.mp4}")
OUT_DIR=$(mkdir -p "${OUT_DIR:-$REPO/outputs}" && cd "${OUT_DIR:-$REPO/outputs}" && pwd)
if [ -z "${PROMPT:-}" ]; then
  if [ "$REF_IMG" != "$(realpath "$DEMO/reference.png")" ]; then
    echo "set PROMPT to a caption of $REF_IMG (README: 人物外观描述… 背景描述…)" >&2; exit 2
  fi
  PROMPT=$(cat "$DEMO/prompt.txt")
fi

if [ "$MODE" = distill ]; then
  CONFIG=wan_animate_2_distillation_v100.yaml; DIT=wan_animate_2_bf16_distillation.safetensors
  STEPS=${STEPS:-10}; GUIDE=${GUIDE:-1.0}            # README: 10 steps, no CFG
else
  CONFIG=wan_animate_2_v100.yaml; DIT=wan_animate_2_bf16.safetensors
  STEPS=${STEPS:-40}; GUIDE=${GUIDE:-3.0}            # demo defaults: 40 steps, CFG 3.0
fi
case "${RES:-480p}" in
  256x448) W=256; H=448 ;;  # experimental 16GB preset only
  288x512) W=288; H=512 ;;  # experimental 16GB preset only
  480p) W=480; H=832 ;;
  540p) W=544; H=960 ;;
  720p) W=720; H=1280 ;;
  [0-9]*x[0-9]*) W=${RES%x*}; H=${RES#*x} ;;  # any WxH (only the area counts), e.g. pad_replace.py --px canvas
  *) echo "RES must be 256x448|288x512|480p|540p|720p|<W>x<H>" >&2; exit 2 ;;
esac

WAN=$CKPT_DIR/videomodel/Wan-AI
for f in "$CKPT_DIR/wan_animate_2/$DIT" "$WAN/models_t5_umt5-xxl-enc-bf16.pth" "$WAN/vae.pth" \
         "$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$WAN/umt5-xxl" "$WAN/xlm-roberta-large" \
         "$REF_IMG" "$DRIVE"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ $((40 % NGPU)) -eq 0 ] || { echo "the 40 attention heads must divide by the GPU count ($NGPU)" >&2; exit 2; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WAN_SDPA_Q_CHUNK=${WAN_SDPA_Q_CHUNK:-4096}
LOG="$OUT_DIR/v100_${MODE}_${RES:-480p}_$(date +%Y%m%d_%H%M%S).log"
echo "MODE=$MODE RES=${RES:-480p} (${W}x${H}) STEPS=$STEPS GUIDE=$GUIDE GPUS=$CUDA_VISIBLE_DEVICES log=$LOG"

cd "$REPO/infer"
# shellcheck disable=SC2086
python wan_animate_2_demo.py \
  --config "./$CONFIG" --dtype "${DTYPE:-auto}" \
  --prompt "$PROMPT" --refer-img-file "$REF_IMG" --refer-video-file "$DRIVE" \
  --width "$W" --height "$H" --fps "${FPS:-24}" --clip_len "${CLIP_LEN:-81}" \
  --sample_guide_scale "$GUIDE" --step "$STEPS" --seed "${SEED:-42}" --output-dir "$OUT_DIR" \
  --cfg-override \
    "model_path=$CKPT_DIR/wan_animate_2/$DIT" \
    "model.t5_checkpoint=$WAN/models_t5_umt5-xxl-enc-bf16.pth" "model.t5_tokenizer=$WAN/umt5-xxl" \
    "model.vae_checkpoint=$WAN/vae.pth" \
    "model.clip_checkpoint=$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "model.clip_tokenizer=$WAN/xlm-roberta-large" \
    "test_cfg.sp_size=$NGPU" "test_cfg.sharding_size=$NGPU" ${EXTRA_OVERRIDES:-} \
  2>&1 | tee "$LOG"
