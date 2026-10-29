# In-context QKV training (skeleton)

`train_incontext_qkv.py` teaches a video DiT to take its motion from a driving video through in-context tokens,
training only the shared self-attention Q/K/V and freezing everything else (the recipe described for
Wan-Animate-2). Reference architecture: the Wan-Animate-2 model in `wanxiang/models/` (its `forward_ref` /
`forward_gen` are the in-context path). Skeleton status: the loop, the freezing, LoRA and the loss are tested on a
tiny model (`tests/test_train_incontext.py`); real training needs a data-prep step that writes the cached samples.

## Training sample (one clip, already encoded)

| key | shape | what |
|---|---|---|
| `target` | C, T, h, w | latents of the clip to reconstruct |
| `ref` | C, 1, h, w | the reference image (character) |
| `drive` | C, T, h, w | the driving clip: another person doing the motion (teacher pair from Wan-Animate-2), or the target clip with its appearance degraded (`augment_driving`, self-reconstruction) to make copying harder |
| `zero` | C, T, h, w | latents of an all-grey clip (unknown frames of the image-to-video condition) |
| `clip_ref`, `clip_drive` | 257, 1280 | CLIP image features of the reference and of the first driving frame |
| `text`, `text_ref` | L, 4096 | T5 of the prompt and of the driving prompt |

## Trainable part

- `--mode qkv`: the full Q/K/V weights of every self-attention (as in the paper).
- `--mode lora --rank 64`: a LoRA on each Q/K/V; `merge_lora` folds it back for the unchanged inference pipeline.

## Porting to another backbone

Rewrite only `model_inputs`, `predict` and `qkv_layers`; keep `flow_loss`, `set_trainable` and `train`.
For a backbone with the opposite time convention (t = 0 noise, x_t = t·x1 + (1−t)·x0, v = x1 − x0) change the two
lines in `flow_loss` that build `x_s` and the target. For an MMDiT, the driving tokens become one more stream in the
joint attention and `qkv_layers` returns the video-stream QKV of the dual- and single-stream blocks.

## Next steps

1. Data prep: clips → (VAE, CLIP, T5) → cached `*.pt` with the keys above; teacher pairs generated with Wan-Animate-2.
2. Multi-GPU (FSDP / sequence parallel) and gradient checkpointing for real model sizes.
3. Evaluation: pose error between driving and output, identity similarity to the reference.

## Pilot on Wan2.2-TI2V-5B (runnable end to end)

TI2V-5B (public diffusers weights) cannot follow a driving video out of the box, so it can show whether the method
itself works, which Wan-Animate-2 cannot (it is already trained for this). Scripts:

| file | does |
|---|---|
| `filter_clips.py` | `score`: Grounding DINO person boxes + motion per clip; `select`: one main person, ≥ 40 % of the frame height, the stillest 35 % dropped |
| `prep_incontext_ti2v5b.py` | 2 windows per clip → cached latents: `target` (latent frame 0 = a reference frame ≥ 1 s from the window) and `drive` (the window zoomed/shifted, grey, blurred); `text.pt` (one generic prompt) |
| `train_incontext_ti2v5b.py` | driving tokens appended to the video tokens (same RoPE positions, own timestep, latents noised to a random level 0.2–0.7), LoRA on the self-attention Q/K/V, fp16 autocast with a float32 residual stream, gradient checkpointing, `torchrun` over the GPUs |
| `infer_incontext_ti2v5b.py` | reference image + driving video → video (`--lora`), or the untrained base with the same input (no `--lora`) |
| `run_pilot_ti2v5b.sh` | all stages on one node: `bash train/run_pilot_ti2v5b.sh` (each stage resumes) |

The conditions are made harder to copy: the reference is another moment of the clip, and the driving clip gets
grey/blur/zoom/shift augmentation and latent noise. Clips without a sufficiently separated reference/window pair
are skipped and counted. Appearance information can remain; held-out cross-identity evaluation is what shows motion
transfer. Compare the same drive with different references, the same reference with different drives, and static or
shuffled driving controls. The `infer` stage writes trained and untrained outputs beside the driving video.

Prep resumes each missing window deterministically and publishes samples and `text.pt` atomically. `manifest.json`
records the cache settings; incompatible settings or an old cache without a manifest require a new cache directory.
Training resumes from atomic `state.pt` checkpoints (LoRA, optimizer, scheduler, scaler, step, sample cursor and
per-rank RNG state). Run settings, including the planned total steps, must match; use a new output directory to
change them. For an early stop in a Python test, `train(..., steps=3, total_steps=6)` retains the six-step plan for
`train(..., steps=6)` to resume. The launcher still skips training when the final LoRA export exists.
AMP logs include scale and skipped updates; only real updates advance the LR schedule, and 50 consecutive skips
stop training. Step/checkpoint numbers count attempted updates.

## P2: teacher pairs (different driving and target person)

Self-reconstruction cannot teach the identity swap (the driving clip shows the target person), so P2 adds pairs made
by Wan-Animate-2, as in its own paper: "person B does person A's motion".

| file | does |
|---|---|
| `make_teacher_jobs.py` | A = a 33-frame window of one clip (written at 24 fps, 480×832), B = a frame of another clip |
| `../infer/wan_animate_2_batch.py` | Wan-Animate-2 move mode over all jobs with one loaded model (resumable) |
| `prep_incontext_ti2v5b.py --teacher jobs.jsonl` | target = [B, teacher frames 1..], drive = A with the self-reconstruction treatment |
| `run_teacher_pairs.sh` | jobs → generate → prep on one node |

Then `RUN=run_p2 DRIVE_ROPE=offset EXTRA_DATA=cache_teacher MIX="0.6 0.4" bash train/run_pilot_ti2v5b.sh train infer`
(`train_incontext_ti2v5b.py --data A B --mix 0.6 0.4` draws from both caches by weight).

## WA2 shared-QKV port to TI2V-5B

`wa2_ti2v5b.py` runs a clean driving pass at embedding timestep 1, then a generation pass through the same blocks.
Generation frame 0 is the clean reference; frames 1–8 retrieve their aligned driving frame in one attention softmax.
Driving positions have a width offset of 15 tokens. Only the 180 self-attention Q/K/V tensors train (849,623,040
parameters). Frozen weights default to fp16, QKV stays fp32, and multi-GPU training shards Adam state with ZeRO.
Both passes checkpoint every block. Full-step V100 memory and learning quality still need hardware validation.

Prep jointly encodes `[reference, clip[1:]]` as the target and encodes the original clip as driving. GDINO/SAM2 person
masks select a deterministic latent Gaussian blur (sigma 2, one-cell mask dilation). No driving noise or pixel
augmentation is added. Frame 0 is conditioning, not reconstruction. The new versioned cache keeps samples under
`samples/`; the pilot and WA2 loaders refuse each other's caches. `REUSE_TARGET=cache` can reuse matching pilot
targets; the drive is always re-encoded. The TI2V-5B weights must be local; GDINO/SAM2 are taken from
`$CKPT_DIR/<name>`, else `<repo>/ckpts/<name>` (as in the pilot's filter stage), else downloaded by HF id. The
last `HOLDOUT=4` clips of `pass.txt` are kept out of training (`wa2_holdout.txt`) for the evaluation below.

Reuse `$WORK/pass.txt` from the pilot's filter/select stages, then run:

```bash
bash train/run_wa2_ti2v5b.sh prep train infer      # defaults: 4x V100, /cache paths as in run_pilot_ti2v5b.sh
```

The fixed layout is 33 frames at 480×832 (9×26×15 tokens per branch). `T5_WAN`/`T5_TOKENIZER` select the existing
local Wan-format text encoder. Prep writes `PREVIEW=4` videos `$WORK/$CACHE/preview_*.mp4` (mask overlay | decoded
blurred drive): check them before training. Training resumes from `state.pt` and exports `qkv_*.safetensors` every `SAVE_EVERY=500` steps (~1 epoch on 4 GPUs).
`SYNC_TO=obs://bucket/path` also copies each save there with ModelArts MoXing, in the background (a failed upload is
logged, training goes on); a fresh machine with an empty `$WORK/$RUN` first fetches `state.pt` from it and resumes.
Inference uses conditional Euler (guide 1) and writes raw and `drive-blurred | output` videos. The launcher's infer
stage tests the latest QKV in-domain, on the held-out clip and the training clip that move most (the pilot filter's
motion score in `$WORK/scores_*.csv`), with the reference taken from the
clip itself 1 s after the window (as in training). Each clip gets four controls: `drive` (its own blurred driving),
`none` (`--no-drive`), `reverse` (`--reverse-drive`) and `wrong` (`--drive-clip` of another clip). Learned motion
means `drive` follows the clip while `reverse`/`wrong` follow their own driving and `none` does not.
