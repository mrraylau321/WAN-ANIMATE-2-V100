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
| `drive` | C, T, h, w | the driving clip: another person doing the motion (teacher pair from Wan-Animate-2), or the target clip with its appearance destroyed (`augment_driving`, self-reconstruction); never the untouched target, the model would just copy it |
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

None of the conditions contains a copyable answer: the reference is another moment of the clip, the driving clip has
no colours or texture, is not pixel-aligned with the target, and is noised in latent space.
Check that it works: the `infer` stage writes the trained and the untrained result side by side with the driving video.
