# Wan-Animate-2 on NVIDIA V100 (sm70): 8×32 GB / experimental 8×16 GB

**Unofficial port.** Wan-Animate-2 (model, paper and original code) is by the Wan-Animate-2 authors;
see the [official repository](https://github.com/Wan-Video/Wan-Animate-2) and [arXiv:2608.06009](https://arxiv.org/abs/2608.06009).
This repository is a minimal patch on top of the official code @ `3ad2fef` that lets the inference
pipeline run on Volta GPUs.

**On sm ≥ 80 (A800/A100/H100, with flash-attn installed), the original compute recipe and tensor
operations are preserved.** Every switch defaults to `auto`, which resolves there to the upstream
bf16 + FlashAttention/flex recipe, for the original YAMLs and the `*_v100.yaml` files alike. The
port-only tensor work (K/V clones, cache frees, extra checks) is gated to the V100 path.

**The launcher and error-handling changes apply everywhere**: worker status reporting,
per-request completion acks, failure propagation and `close()`. There are also small Python-side
dispatch checks.

Engineering details, the blocker inventory, the review responses, test results and risks are in
[NOTES_V100.md](NOTES_V100.md).

> **Status: CPU-verified only.** 107 CPU tests (including 16 GB shard/FP16-forward and launcher checks) cover the following, compared against the unmodified
> upstream code and the real checkpoint headers:
> * the attention re-implementation;
> * the loader and launcher paths;
> * the fp16 checks.
>
> **No GPU run has been done yet.** See NOTES §11 for the V100 checklist.

**BF16 on V100, in one line:** `torch.cuda.is_bf16_supported()` says True on V100 only through
*emulation*. V100 has no bf16 tensor cores, and SDPA's fused kernels reject bf16 below sm80 (bf16
falls back to the MATH kernel and its full N×N score matrix). So this port computes in **fp16**,
keeps fp32 where range matters, and checks for overflow (see "fp16 safety" below). Run
`scripts/probe_v100.py` to see it on your card.

## What changed and why

| change | why |
|---|---|
| `wanxiang/models/attention_sdpa.py` (new); dispatch in `attention.py` | FlashAttention 2/3 and Triton `flex_attention` do not run on sm70. PyTorch SDPA replaces both with the same semantics: flash varlen `k_lens` (padding sliced away); the flex BlockMask + `score_mod` evaluated from the *same* `mask_mod`, with per-frame runs of keys. Fused kernels are used first; if none accepts the inputs, the MATH kernel runs with memory-bounded query chunks. |
| `wan_animate_2_model.py`: `create_mask` closure moved to `incontext_mask_mod()`, one `if`, cached K/V cloned | Lets the SDPA path reuse the identical mask logic. The cached K/V own their storage, so the Q buffer is not kept alive. No parameter keys changed. |
| `wanxiang/utils/precision.py` (new); pipeline dtype plumbing | sm < 80 gets fp16 DiT (fp32 residual between blocks), fp32 T5 and VAE, fp16 CLIP. Overrides: `--dtype`, `WAN_DTYPE`, YAML. |
| `wanxiang/utils/fp16_guard.py` (new) | fp16 tops out at 65504 (bf16 at ~3.4e38). Adds load-time range checks and runtime inf/NaN detection at the attention kernel boundary and the model output, with per-block fp32 fallback or abort. |
| `pipelines/wan_animate_2_pipeline.py` | See the list below this table. |
| `wanxiang/distributed/fsdp.py` | `extra_wrap`: on V100 the T5 token embedding and final norm become their own FSDP units, so FSDP1 does not keep a 4.2 GB unsharded root after encoding. |
| `core/__init__.py`, `infer/wan_animate_2_demo.py` | `--dtype`, `--attn-backend`, `--cfg-override key=value ...`; the demo stops its workers at exit (upstream's would hang). |
| `infer/wan_animate_2{,_distillation}_v100.yaml` | V100 knobs, all `auto` (only `test_cfg` differs from upstream). |
| `scripts/` | `run_v100_base.sh`, `run_v100_distill.sh` (→ `run_v100.sh`), `probe_v100.py`, `check_ckpt_fp16_range.py`, `estimate_v100_memory.py` |
| `tests/` | CPU tests (see NOTES §8) |

Changes in `pipelines/wan_animate_2_pipeline.py`:

* precision switch;
* DiT build/load moved into `build_dit()` (same statements, now testable);
* `t5_dtype=fp16` is refused (umT5 overflows fp16 and its clamp would hide it; fp32 or bf16 only);
* CLIP loaded in fp32, checked, then cast;
* fp16 load checks;
* guard around every model forward;
* collective inf/NaN checks on CLIP, T5 and VAE outputs;
* VAE caches and the K/V cache freed after use;
* a request succeeds only after rank 0's result **and** a completion ack from every worker; any
  reported failure is latched and a worker death is detected, so the launcher raises the original
  error and exits non-zero instead of hanging or returning a partial success;
* refuses `torchrun --nproc_per_node>1`.

## Weights

The official release is **BF16 only**: Base (40 steps, CFG 3.0) and Distillation (10 steps, no CFG),
each a 32.8 GB safetensors with 1303 tensors and 16.39 B params. **There is no official FP16 file.**
This port casts BF16 → FP16 *after* the strict load, with range checks. Community variants
(Comfy-Org bf16/int8, GGUF quantisations, the diffusers repos) are listed in NOTES §2; this code
does not use them.

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir ./ckpts   # inside this repo
```

## Install (V100 server)

Follow the [upstream README](https://github.com/Wan-Video/Wan-Animate-2#readme), with two differences:

* **Skip `flash-attn`.** It cannot run on Volta.
* Keep the **cu126** (or cu118) torch wheels: `torch==2.7.1+cu126`. Newer CUDA builds may not ship
  sm_70 kernels; `probe_v100.py` prints `torch_arch_list`.

```bash
conda create -n wan_animate_2 python==3.11 -y && conda activate wan_animate_2
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt && pip install -e .
# ffmpeg must be on PATH (audio extraction / muxing)
```

## Run

```bash
python scripts/probe_v100.py --json probe.json                    # 1. what the card supports
python scripts/check_ckpt_fp16_range.py \
    ckpts/wan_animate_2/wan_animate_2_bf16.safetensors \
    ckpts/wan_animate_2/wan_animate_2_bf16_distillation.safetensors   # 2. fp16 pre-flight (streams, low RAM)
python scripts/estimate_v100_memory.py                            # 3. per-phase memory estimate (CPU)
RES=480p scripts/run_v100_distill.sh                              # 4. fastest end-to-end check
RES=480p scripts/run_v100_base.sh                                 # 5. Base, 40 steps, CFG 3.0
RES=720p scripts/run_v100_base.sh                                 # 6. only after 480P/540P were measured
```

By default the scripts use the upstream example `examples/demo1/` (`reference.png`, `template.mp4`,
and the caption in `prompt.txt`), so a fresh clone runs out of the box. The demo video is 337 frames
at 30 fps, which gives several 81-frame clips.

For your own inputs, set `REF_IMG`, `DRIVE` and `PROMPT`. `PROMPT` is a caption of `REF_IMG` in the
[upstream README](https://github.com/Wan-Video/Wan-Animate-2#readme)'s `人物外观描述… 背景描述…` format, and it is required when `REF_IMG` changes. Example:

```bash
# REF_IMG=/data/my_ref.jpg DRIVE=/data/my_drive.mp4 \
# PROMPT="人物外观描述：一个关节可动的机器人人偶，白色的脸部，蓝色的眼睛，深棕色的头发在头顶盘成发髻。身体由黑色和深灰色的分段式关节部件组成，腰部有一圈红色的装饰带，手臂为银白色与黑色相间的机械关节，脚穿白色运动鞋。背景描述：背景为一块米色的亚麻布幕，地面铺着灰色的方形石砖，光线柔和自然。" \
# RES=480p scripts/run_v100_distill.sh
```

Other knobs:

* `RES` = `480p` | `540p` | `720p` (portrait 480×832, 544×960, 720×1280)
* `STEPS`, `GUIDE`, `CLIP_LEN` (81), `FPS` (24), `SEED`, `GPUS` (0–7)
* `DTYPE` = `auto` | `fp16` | `fp32` | `bf16`
* `CKPT_DIR`, `OUT_DIR` (default `outputs/`)
* `EXTRA_OVERRIDES="test_cfg.overflow_action=abort"`

The output is written to `outputs/session_*/results.mp4` together with a log file.

**Launching.** The pipeline spawns one worker per visible GPU by itself, as upstream does. Start
**one** `python` process per node. `torchrun --nproc_per_node=8` would create 64 ranks and is
refused.

**On failure** (a rejected checkpoint, an fp16 error that has to abort, a crashed worker, a worker
failing after rank 0 already produced its result), the launcher stops all workers. It exits
non-zero with the original error, for example
`Wan-Animate-2 initialisation failed: worker 0: FP16OverflowError: …`. It does not hang.

A request is only reported as successful when every worker has acknowledged finishing it and all
workers are still alive. `WAN_INIT_TIMEOUT_SEC` and `WAN_REQUEST_TIMEOUT_SEC` add optional hard
timeouts.

### Equivalent manual command
```bash
cd infer && PYTHONPATH=.. python wan_animate_2_demo.py --config ./wan_animate_2_distillation_v100.yaml \
  --dtype auto --prompt "$(cat ../examples/demo1/prompt.txt)" \
  --refer-img-file ../examples/demo1/reference.png --refer-video-file ../examples/demo1/template.mp4 \
  --width 480 --height 832 --sample_guide_scale 1.0 --step 10 --seed 42 \
  --cfg-override model_path=../ckpts/wan_animate_2/wan_animate_2_bf16_distillation.safetensors
```

## Experimental **8× V100 16 GB** launch presets

The original **32 GB launchers, their default FP32 DiT shard storage, and their default
480P/540P/720P behavior remain unchanged**. The 16 GB path is opt-in and still
requires **eight separate V100 16 GB GPUs** using the model's FSDP2 and
context-parallel implementation. It is **not** a 1×16 GB or 2×16 GB model port.

| `PRESET` | Width × height | Chunk frames | Intended use |
|---|---|---:|---|
| `minimal` (default) | 256×448 | 17 | first low-memory smoke test |
| `balanced` | 256×448 | 25 | longer context at same resolution |
| `detail` | 288×512 | 25 | more spatial detail; more memory |
| `experimental480` | 480×832 | 17 | **high OOM risk**, explicit opt-in only |

The above names indicate configurations, **not certified OOM-free tiers**. All are
unverified on actual 16 GB cards; even the minimal preset might OOM. The default
8-way allocation estimates **3.82 GiB DiT BF16 shards + 0.04 GiB FP32
island shard overhead + 0.10 GiB remaining DiT root + 1.32 GiB T5 shard +
0.15 GiB CLIP shard + 0.47 GiB replicated VAE per GPU**, or about **5.90 GiB
persistent weights**. The FP32 islands are timestep embedding, timestep
projection and output head (about 185M parameters, sharded 8-way). Dynamic peaks also include the approximately
4.40 GiB T5 FP32 embedding gather, approximately 1.50 GiB DiT prefetch, and
reference K/V and VAE activations (phase-dependent, not all simultaneous).
The offline estimator gives **11.85 / 11.87 / 11.89 GiB** peak per GPU for
minimal / balanced / detail, respectively, **including an assumed 1.5 GiB
untraced overhead**; actual peaks may be higher and depend on the V100 server.
**Host memory is also required**: the current 8-worker loader may temporarily use
roughly 180–210 GiB of system RAM while loading T5, even on 16 GB GPUs.
The 480P option is gated behind `ALLOW_UNVERIFIED_HIGH_RES=1`; do not interpret a
successful 32 GB run as evidence it fits 16 GB. `MODE=distill` (10 steps, no CFG)
is the launch default, or use `MODE=base` (40 steps + CFG).
Resolution and chunk length may affect **output quality and chunk-seam consistency**;
the shortest clip is not guaranteed to give equivalent motion.

```bash
python scripts/probe_v100.py
python scripts/check_ckpt_fp16_range.py \
  ckpts/wan_animate_2/wan_animate_2_bf16_distillation.safetensors
python scripts/estimate_v100_memory.py --profile 16gb --res 256x448 --clip-len 17
python scripts/estimate_v100_memory.py --profile 16gb --res 256x448 --clip-len 25
python scripts/estimate_v100_memory.py --profile 16gb --res 288x512 --clip-len 25
# These estimates miss CUDA contexts, NCCL workspaces, caching/fragmentation;
# supply the actual device capacity with --capacity-gib.

DRY_RUN=1 PRESET=minimal bash scripts/run_v100_16gb.sh
PRESET=minimal  bash scripts/run_v100_16gb.sh
PRESET=balanced bash scripts/run_v100_16gb.sh
PRESET=detail   bash scripts/run_v100_16gb.sh
MODE=base PRESET=minimal bash scripts/run_v100_16gb.sh

# DANGEROUS EXPERIMENT, not a supported 480P fit claim:
PRESET=experimental480 ALLOW_UNVERIFIED_HIGH_RES=1 bash scripts/run_v100_16gb.sh
```

The released checkpoint is **BF16**, not FP16. On the 16 GB path,
`test_cfg.dit_shard_dtype=bf16` retains original checkpoint numeric values in
the large-block FSDP2 shards (roughly **3.8 GiB/GPU** less than the 32 GB
recipe). The timestep embedding/projection and output head require separate
FP32 FSDP2 islands to avoid mixed-dtype failures; the launcher applies
`test_cfg.fp32_modules=[time_embedding,time_projection,head]`. FSDP2 **computes** with guarded FP16 on V100,
not emulated BF16; the residual between blocks, T5 and VAE are FP32. As
another memory reduction, only the decoded overlap frame is carried to the
next chunk on CPU (`low_vram=true`). The VAE is cast to FP32 on CPU **before**
GPU transfer to avoid holding duplicate BF16/FP32 VAE weights on the card.
`log_gpu_memory=true` logs actual CUDA allocated/reserved peaks and **local
weight allocation by dtype** for DiT, T5, CLIP, and VAE on **every rank**.
The 16 GB launcher defaults to `overflow_action=abort` rather than risking an
unplanned FP32 block fallback that might OOM; an FP32 island can also increase
persistent shard memory. The launcher refuses FP16 **checkpoint storage**.

**Important: running in FP16 is not numerical/visual correctness.** The existing
inf/NaN guards detect some unsafe activations, not finite-valued degradation.
On real V100 GPUs, record all eight peak-reserved values and confirm no OOM,
then compare identical inputs, prompts, seeds, resolution and `CLIP_LEN`
against a high-precision FP32-compute reference when feasible on 32 GB
cards. Review both the whole video and **chunk boundaries** for brightness /
color shifts, flicker, lost identity and motion discontinuities; check
per-frame RGB/exposure and seam transitions. Also validate both Base and
Distillation. Until those runs pass, this profile must stay labelled
**experimental / GPU-unverified**.

---

## Expected limits — original 32 GB recipe (estimates, not measurements; see NOTES §7)

The figures below are for **one 81-frame clip** of a 64-frame driving video (65 frames after
resampling and padding), on 8× V100 32 GB. Memory is GiB per GPU, including 1.5 GiB of untraced
overhead, against 31.75 GiB capacity. It is the peak over the run phases (conditioning, reference
pass, denoising, VAE decode). The transient peaks of the VAE and of one DiT block are traced with
FakeTensors; everything else is counted analytically.

| | 480P | 540P | 720P |
|---|---|---|---|
| peak phase | VAE decode ~18.6 GiB | VAE decode ~20.7 GiB | VAE decode ~27.7 GiB (**tight**) |
| denoising (default / MATH fallback / a block in fp32) | 17.7 / 17.9 / 19.0 | 18.8 / 19.3 / 20.3 | 22.7 / 23.2 / 24.5 |
| Distillation (10 forwards), NVLink / PCIe | ~0.8 / ~1.3 min | ~1.2 / ~1.8 min | ~3.3 / ~3.9 min |
| Base (80 forwards), NVLink / PCIe | ~6 / ~10 min | ~10 / ~14 min | ~26 / ~31 min |

**720P is conditional on measurement.** fp32 VAE decode (~16 GiB transient) and encode (~13 GiB)
take it to 26–28 GiB of the 31.75 GiB available. Allocator fragmentation is not modelled. Run 480P
and 540P first, compare `torch.cuda.max_memory_allocated` with this table, and only then try 720P.

The times are FLOP-based guesses: 40 TFLOPS attention and 80 TFLOPS GEMM, NVLink at ~100 GB/s or
PCIe at ~10 GB/s. They exclude model loading (minutes), T5 and the VAE.

**Host RAM** (estimate, not measured) peaks at **~180–210 GB** during initialisation. Each of the
8 workers holds the bf16 T5 module plus its loaded state dict (~22.7 GB) at the same time, and
upstream behaves the same.

The port loads CLIP in fp32 (~9.5 GB per worker, against 7.2 GB upstream) so it can check it
before the fp16 cast. That load happens after T5 has moved to the GPU, so it does not raise this
peak.

## fp16 safety: exactly what is checked

| check | when | what fails | action |
|---|---|---|---|
| DiT load check (`[fp16-check] DiT`) | once, after the strict load, on the GPU shards (all-reduced) | any non-finite weight; any **original** \|w\| > 65504; more than `fp16_underflow_frac` (default 0.001) of a module's nonzero weights becoming 0 in fp16 (\|w\| ≤ 2^-25 ≈ 2.98e-8) | raises and names the modules, unless they are in `test_cfg.fp32_modules` (kept and run in fp32). **Every** flushed weight is listed as a WARNING with count and max \|w\|, even within the tolerance. fp16 subnormals (\|w\| < 2^-14) are reported only. |
| CLIP load check (`[fp16-check] CLIP visual`) | CLIP is strictly loaded **in fp32**, checked, then cast to fp16 | same rules | raises; fix: `test_cfg.clip_dtype=fp32` |
| pass markers (`WAN_FP16_CHECK`) | after each passed load check | — | a pass is recorded under `~/.cache/wan_animate2_v100/fp16_check` (or `WAN_FP16_CHECK_CACHE`), keyed by the weight file's path, size and mtime plus `fp32_modules` / `fp16_underflow_frac`. `auto` (default): the same file with the same settings skips the check (CLIP's takes minutes on CPU), anything changed is checked again; `force`: always check; `off`: never check (logged as a WARNING). A failed check is never recorded. The runtime monitors below are unaffected. |
| `scripts/check_ckpt_fp16_range.py` | before the first run, on the server | the same functions and thresholds | PASS/FAIL, the flush list, and a suggested `fp32_modules: [...]` |
| T5 dtype | at start-up | `t5_dtype=fp16` (or `param_dtype=float16` on the upstream recipe) | refused with an error: umT5 is known to overflow fp16 and its `fp16_clamp` would turn inf into finite values; use fp32 (V100 default) or bf16 |
| attention kernel monitor | every DiT forward, every step | inf/NaN in q/k/v/out of **every** SDPA call (post-RoPE, post-cast, post-slicing). This catches overflow that attention would otherwise hide (an all-`-inf` logit row returns 0). | device-side flag, read once per forward |
| model-output sentinel | every DiT forward, every step | inf/NaN in the forward's output | same |
| detailed hooks (`overflow_guard: first_step`, the V100 default, or `all`) | reference pass + step 0 of every clip; any step that tripped a flag is re-run with hooks | first inf/NaN in execution order among: block outputs, `pre_self_attention` (q/k/v + six modulation chunks), raw attention output, cross-attention, FFN up-projection and output, root modules, and every attention kernel tensor (labelled) | `overflow_action: fp32` (default) switches that **block** to fp32 and recomputes the same forward; `abort` stops. WARNING when max\|x\| ≥ 16384. |
| conditioning checks | every clip | inf/NaN in VAE latents, CLIP outputs (including CLIP's attention kernels), T5 contexts, and VAE decoded frames before the [-1, 1] clamp | all ranks raise the same error (no fallback) |

Every decision is made **collectively**, so all ranks fall back or abort together. Nothing raises
inside a hook or a kernel.

Collectives per guarded DiT forward:

* non-detailed step: **one all-reduce of 2 floats** (output sentinel + kernel flag) and one
  device→host readback;
* detailed step: **two all-reduces**, the record counts as max and −min (so a mismatch aborts every
  rank) and then the packed flags and values, plus two readbacks;
* a re-run adds another forward.

The conditioning checks add one all-reduce and one readback per group: VAE+CLIP, T5 (one or two),
and the VAE decode.

**Not checked:** finite-but-degraded results that never reach inf/NaN. That covers fp16 rounding
drift and large finite values that are later normalised. Values near the limit produce WARNINGs on
detailed steps. An fp32 reference run (`--dtype fp32`) is the way to compare quality.

**Overhead.** The monitor adds 4 min/max reductions per attention call. On detailed steps, about
15 more reductions per block, plus the collectives above. This is **not measured on GPU**; it is
expected, not shown, to be small next to a 5–20 s forward. Set `overflow_guard: all` to hook every
step, or `off` to disable all of the runtime checks.

**Example log lines** (verbatim from CPU tests with injected values; `…` marks elided parts):
```
WARNING [fp16-check] DiT: 1 nonzero weights become 0 in fp16 across 1 modules (all |w| <= 2.98e-8):
  m: 1/2000 (5.00e-04) flushed, max flushed |w|=1e-09
ERROR [fp16-guard] non-finite (inf/NaN) activation at step=3 branch=cond: first offending module <attention kernel input/output> (not localised) (dtype=); re-running this forward with hooks to localise
ERROR [fp16-guard] non-finite (inf/NaN) activation at step=3 branch=cond: first offending module blocks.0.block.cross_attn:attn0.q (in blocks.0) (dtype=float16)
ERROR [fp16-guard] FALLBACK: 'blocks.0' now runs in fp32 activations for the rest of this run (fp32 units: ['blocks.0']); re-running step=3 branch=cond
```

How to read the labels:

* `module[i]` is the i-th tensor output of a module.
* `:method` names the `AttentionBlock` call:
  * `pre_self_attention` returns q, k, v and then the six modulation chunks;
  * `post_self_attention:attn_out` is the raw attention output.
* Kernel tensors are labelled `<module>:<attn|masked_attn><n>.<q|k|v|out>`:
  * `blocks.N:forward_ref:attn0.*` is the reference self-attention;
  * `blocks.N:forward_gen:masked_attn0.*` is the generation (In-Context) self-attention;
  * `blocks.N.block.cross_attn:attn0.*` is the image-context cross-attention, and `attn1.*` the
    text cross-attention;
  * a trailing `(in blocks.N:forward_gen)` names the enclosing block call.
* `step=ref` is the driving-video reference pass; `step=N` is the sampling step.
* A `FALLBACK` line means the result is recomputed in fp32 for that block, which is slower. If the
  same block falls back on every run, add it to `fp32_modules`.

## Known differences vs the official A800 path

1. fp16 instead of bf16 compute, so expect small numeric drift. The residual stream between blocks
   is fp32 here.
2. T5 and VAE run in fp32 (upstream: bf16), and the pixel tensors fed to the VAE are fp32. T5 stores
   its weights in bf16 as released and computes in fp32. fp16 T5 is refused.
3. SDPA instead of FlashAttention/flex. The maths is identical up to rounding. The distillation
   `log_scale` bias is rounded to fp16 (−1.2998 vs −1.3).
4. Slower: V100 peaks at ~125 fp16 tensor TFLOPS (SXM2) vs 312 bf16 TFLOPS on A800, kernels reach
   a lower fraction of that peak, and FSDP gathers parameters on every forward.

## Attribution and license

Wan-Animate-2 is © the Wan-Animate-2 authors; see the [upstream repository](https://github.com/Wan-Video/Wan-Animate-2) and the
[project page](https://humanaigc.github.io/wan-animate-2). It is licensed under the
[Apache License 2.0](LICENSE), and the original `LICENSE` is unchanged.

This V100 port (October 2026) is a derivative work, also under Apache-2.0:

* modified upstream files carry a "Modified for the V100 (sm70) port" notice at the top;
* new files are marked in their docstrings.

Paper: *Wan-Animate-2*, arXiv:2608.06009. Model weights: Hugging Face `Wan-AI/Wan2.2-Animate-2-14B`.
