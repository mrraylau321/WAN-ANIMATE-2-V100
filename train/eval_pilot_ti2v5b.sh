#!/usr/bin/env bash
# Diagnostic grid for a trained pilot LoRA: tells sampler problems, out-of-domain inputs and copying apart.
#   bash train/eval_pilot_ti2v5b.sh            -> $WORK/eval/<name>_side.mp4 (driving | generated), 7 runs on the GPUs
# Runs (name: what it answers):
#   1 base_nodrive_g1   plain TI2V-5B image-to-video from the demo cat, our Euler sampler, no CFG: is the sampler OK?
#   2 base_nodrive_g5   same with CFG 5 (TI2V's usual setting): does the blur come from missing CFG?
#   3 lora_cat_g5       the demo test again with CFG 5
#   4 lora_self_g1      in-domain: reference = a frame of clip A (5 s in), driving = clip A: can it rebuild a human clip?
#   5 lora_cross_g1     cross-identity humans: reference = a frame of clip B, driving = clip A: motion from A, look of B?
#   6 base_cross_g1     run 5 without the LoRA (what the untrained model does with the same input)
#   7 lora_cross_g5     run 5 with CFG 5
# Clips A and B: the first two clips of $WORK/pass.txt (training clips; the reference/driving pairing of 5-7 is new).
# Knobs (env): WORK CKPT_5B GPUS LORA CLIP_A CLIP_B REF DRIVE FRAMES WEIGHTS
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-/home/ma-user/anaconda3/envs/wan_animate_2}
[[ -d $ENV_PREFIX/bin ]] && export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
VIDEOS=${VIDEOS:-/cache/humanvid_real/vertical_static}
CKPT_5B=${CKPT_5B:-/cache/WAN2.2-TI2V-5B-Diffusers/WAN2.2-TI2V-5B-Diffusers}
WORK=${WORK:-/cache/pilot_ti2v5b}
GPUS=${GPUS:-0,1,2,3}
FRAMES=${FRAMES:-33}
REF=${REF:-$REPO/examples/demo1/reference.png}
DRIVE=${DRIVE:-$REPO/examples/demo1/template.mp4}
LORA=${LORA:-$(ls "$WORK"/run/lora_*.safetensors 2>/dev/null | tail -1 || true)}
[[ -n $LORA ]] || { echo "no LoRA in $WORK/run (set LORA=...)" >&2; exit 1; }
CLIP_A=${CLIP_A:-$VIDEOS/$(sed -n 1p "$WORK/pass.txt")}
CLIP_B=${CLIP_B:-$VIDEOS/$(sed -n 2p "$WORK/pass.txt")}
OUT=$WORK/eval
mkdir -p "$OUT"
cd "$REPO"
IFS=, read -r -a GPU <<< "$GPUS"
echo "LoRA $LORA | clip A $CLIP_A | clip B $CLIP_B -> $OUT"

# reference frames 5 s into each clip (the driving window starts at 0 s, so A's reference is another moment)
python - "$CLIP_A" "$OUT/refA.png" "$CLIP_B" "$OUT/refB.png" <<'EOF'
import sys, cv2
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    cap = cv2.VideoCapture(src)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(5 * fps), n - 1))
    ok, f = cap.read()
    assert ok, f"cannot read {src}"
    cv2.imwrite(dst, f)
    print(f"{dst} <- {src} @ {min(5.0, (n - 1) / fps):.1f} s")
EOF

run() {  # run <name> <gpu> <args...>
  local name=$1 gpu=$2
  shift 2
  CUDA_VISIBLE_DEVICES=$gpu python train/infer_incontext_ti2v5b.py --model "$CKPT_5B" --text "$WORK/cache/text.pt" \
    --frames "$FRAMES" --out "$OUT" --name "$name" ${WEIGHTS:+--weights "$WEIGHTS"} "$@" > "$OUT/$name.log" 2>&1 \
    && echo "ok   $name" || echo "FAIL $name (see $OUT/$name.log)"
}
jobs=(
  "base_nodrive_g1|--ref $REF --drive $DRIVE --no-drive"
  "base_nodrive_g5|--ref $REF --drive $DRIVE --no-drive --guide 5"
  "lora_cat_g5|--lora $LORA --ref $REF --drive $DRIVE --guide 5"
  "lora_self_g1|--lora $LORA --ref $OUT/refA.png --drive $CLIP_A"
  "lora_cross_g1|--lora $LORA --ref $OUT/refB.png --drive $CLIP_A"
  "base_cross_g1|--ref $OUT/refB.png --drive $CLIP_A"
  "lora_cross_g5|--lora $LORA --ref $OUT/refB.png --drive $CLIP_A --guide 5"
)
i=0
for j in "${jobs[@]}"; do
  # shellcheck disable=SC2086
  run "${j%%|*}" "${GPU[$((i % ${#GPU[@]}))]}" ${j#*|} &
  i=$((i + 1))
  (( i % ${#GPU[@]} == 0 )) && wait
done
wait
ls "$OUT"/*_side.mp4
