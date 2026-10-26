#!/usr/bin/env bash
# EXPERIMENTAL, UNOFFICIAL: the training-free replace recipe on Wan2.2-TI2V-5B (public diffusers weights, see
# scripts/ti2v5b_replace.py). New script; the Wan-Animate-2 launchers are untouched.
#
#   CKPT_5B=<Wan2.2-TI2V-5B-Diffusers dir> DRIVE=<video> REF_IMG=<character image> REF_TARGET=cat \
#   PROMPT="..." bash scripts/run_ti2v5b_replace.sh
#
# Weights (34 GB): huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir $CKPT_5B
# Needs diffusers >= 0.35 (pip install -U 'diffusers>=0.35').
# Optional: TARGET=person (what to replace)  REF_PICK=largest|union  GPUS=0,1 (text encoder on the 2nd GPU)
#           STRENGTH=0.6 STEPS=40 GUIDE=5 WIDTH=480 HEIGHT=832 FRAMES=121 FPS=24 DILATE=15 GROW=1 FREE=2
#           DTYPE=fp16|fp32 SEED=42 NO_BLEND=1 OUT_DIR=~/ray_runs/ti2v5b WORK_DIR=inputs
# Other TI2V-5B-architecture weights (CKPT_5B still gives configs, VAE, text encoder, scheduler):
#           TRANSFORMER_CKPT=<one .pth/.safetensors>   or   EXPERTS="<file>@<lo>-<hi> <file>@<lo>-<hi> ..."
#           (timestep ranges, t = 1000 * sigma, together covering 0-1000; one expert on the GPU at a time)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
command -v python >/dev/null || { echo "python not found: activate the conda env first" >&2; exit 1; }
: "${CKPT_5B:?set CKPT_5B=<Wan2.2-TI2V-5B-Diffusers dir>}"
: "${DRIVE:?set DRIVE=<driving video>}"
: "${REF_IMG:?set REF_IMG=<reference image>}"
for f in "$CKPT_5B/transformer/config.json" "$CKPT_5B/vae/config.json" "$CKPT_5B/text_encoder/config.json" \
         "$CKPT_5B/tokenizer/tokenizer_config.json" "$CKPT_5B/scheduler/scheduler_config.json" "$DRIVE" "$REF_IMG"; do
  [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }
done
python - <<'EOF' || { echo "needs diffusers >= 0.35: pip install -U 'diffusers>=0.35'" >&2; exit 1; }
import diffusers
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: F401
v = tuple(int(x) for x in diffusers.__version__.split(".")[:2])
assert v >= (0, 35), diffusers.__version__
EOF

export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
GPU0=0
WORK_DIR=${WORK_DIR:-$REPO/inputs}
mkdir -p "$WORK_DIR"
TARGET=${TARGET:-person}
GDINO=${GDINO:-${CKPT_DIR:-$REPO/ckpts}/grounding-dino-base}; [[ -d $GDINO ]] || GDINO=IDEA-Research/grounding-dino-base
SAM2=${SAM2:-${CKPT_DIR:-$REPO/ckpts}/sam2.1-hiera-large}; [[ -d $SAM2 ]] || SAM2=facebook/sam2.1-hiera-large

# 1) person masks of the driving video + g0 = the reference character on the clip's clean plate
MASK=${MASK:-$WORK_DIR/$(basename "${DRIVE%.*}")_gsam2_$(echo "$TARGET" | tr -c 'A-Za-z0-9\n' '_')_mask.npz}
[[ -f $MASK ]] || CUDA_VISIBLE_DEVICES=$GPU0 python scripts/grounded_sam2.py video --video "$DRIVE" --target "$TARGET" \
  --out "$MASK" --gdino "$GDINO" --sam2 "$SAM2"
SCENE=$WORK_DIR/$(basename "${REF_IMG%.*}")_scene_${REF_PICK:-largest}.png
python scripts/grounded_sam2.py scene-ref --ref "$REF_IMG" --ref-target "${REF_TARGET:-character}" \
  --pick "${REF_PICK:-largest}" --video "$DRIVE" --masks "$MASK" --out "$SCENE" --gdino "$GDINO" --sam2 "$SAM2"

# 2) TI2V-5B: SDEdit from the driving latents + g0 as first frame + background pinned to the driving video
OUT_DIR=${OUT_DIR:-$HOME/ray_runs/ti2v5b}
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/ti2v5b_$(date +%Y%m%d_%H%M%S).log"
# shellcheck disable=SC2046
python scripts/ti2v5b_replace.py --model "$CKPT_5B" --drive "$DRIVE" --g0 "$SCENE" --masks "$MASK" \
  --prompt "${PROMPT:-}" --width "${WIDTH:-480}" --height "${HEIGHT:-832}" --frames "${FRAMES:-121}" \
  --fps "${FPS:-24}" --steps "${STEPS:-40}" --strength "${STRENGTH:-0.6}" --guide "${GUIDE:-5}" \
  --dilate "${DILATE:-15}" --grow "${GROW:-1}" --free-steps "${FREE:-2}" --dtype "${DTYPE:-fp16}" \
  --seed "${SEED:-42}" --out "$OUT_DIR" $([[ ${NO_BLEND:-0} == 1 ]] && echo --no-blend) \
  ${TRANSFORMER_CKPT:+--transformer-ckpt "$TRANSFORMER_CKPT"} $(for e in ${EXPERTS:-}; do echo --expert "$e"; done) \
  2>&1 | tee "$LOG"
