# NOTES_V100 — porting Wan-Animate-2 inference to 8× V100 (sm70, 32 GB)

Engineering notes for review. Public material only. **Nothing here has been run on a GPU.** All
verification is CPU-only: tiny configs, real-module FakeTensor traces, and the real release
checkpoint *headers*. §11 lists what still needs a real V100 run. §13 answers the independent code
review: round 1 (F1–F8) and round 2 (N1–N5, R4).

## 1. Baseline

| item | value |
|---|---|
| upstream code | `github.com/Wan-Video/Wan-Animate-2` @ `3ad2fef7d61d6200c9c653e0fe47be7616b323f3` (2026-08-08, "Fix author list in README.md"), Apache-2.0 |
| weights | HF `Wan-AI/Wan2.2-Animate-2-14B` @ revision `6e8f1973bf0abc2aafd517992e8b6d88c3c46e69` (lastModified 2026-08-09) |
| paper | arXiv:2608.06009 |
| upstream stack | torch 2.7.1+cu126, triton 3.3.1, flash-attn (FA2/FA3), diffusers 0.36.0 (`requirements.txt`) |
| CPU test env | `../.venv-cpu` (uv, Python 3.11.14, torch 2.7.1 CPU, torchvision 0.22.1, diffusers 0.36.0, transformers 4.57.6) |

## 2. Released weights (no download; HF API listing plus safetensors headers fetched with HTTP Range)

| file (HF repo above) | bytes | what it is |
|---|---|---|
| `wan_animate_2/wan_animate_2_bf16.safetensors` | 32,789,901,704 | **Base** DiT: 1303 tensors, **all BF16**, 16,394,878,784 params |
| `wan_animate_2/wan_animate_2_bf16_distillation.safetensors` | 32,789,901,704 | **Distillation** DiT: identical names, shapes and dtypes (BF16) |
| `videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | umT5-XXL encoder, bf16 (5.681 B params by the model definition) |
| `videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | 4,772,359,047 | CLIP ViT-H/14 (0.632 B) + XLM-R text tower (0.561 B). The size matches fp32. |
| `videomodel/Wan-AI/vae.pth` | 783,806,166 | Wan VAE, read as `["model_state"]` (0.127 B params by definition). Its dtype and extra contents are not verified without a download. |
| tokenizers (`umt5-xxl/`, `xlm-roberta-large/`) | small | |

* **There is no official FP16 file.** Both DiT checkpoints are BF16 only. The requested "FP16
  distil" does not exist as an official artifact: the Distillation model is released in **BF16**.
  This port makes fp16 **at load time**: strict load into the upstream fp32 FSDP shards, then a
  per-forward cast to fp16, guarded by the range checks in §6.4.
* The keys are already in the `blocks.N.block.*` layout, so they load directly with
  `set_model_state_dict(strict=True)`. `load_from_official_state_dict` is not used by the pipeline.
  The key/shape/dtype set and the HF revision are pinned offline in
  `tests/fixtures/wan_animate_2_ckpt_header.json` (no weights).
* Diffusers repos `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` and `…-Distilled-Diffusers`: transformer
  in 4 shards totalling about 32.79 GB (bf16), text encoder about 11.36 GB, image encoder 1.26 GB,
  VAE 0.51 GB.
* Community packaging, all for ComfyUI or GGUF runtimes, **none of them fp16 DiTs**:
  * `Comfy-Org/Wan-Animate-2`: `wan_animate_2_bf16`, `wan_animate_2_distill_bf16`, `*_int8_convrot` (16.65 GB), `umt5_xxl_fp16` / `fp8_e4m3fn_scaled` text encoders, `clip_vision_h`, `Wan2_1_VAE_bf16`, and a lightx2v LoRA.
  * `realrebelai/Wan-Animate-2_GGUFs`: Distilled "TURBO" Q2_K…Q8_0 (6.5–18.1 GB).

## 3. Inference path (upstream, end to end)

`infer/wan_animate_2_demo.py` → `core.build_object_from_config_file(yaml)` → `WanAnimate2MPIPipeline.setup()`
→ `main()` **mp.spawn's one worker per visible GPU** (global rank = node_rank × gpus + gpu). It is
not torchrun-style: launch one process per node.

Each worker does the following:

1. NCCL process group and context-parallel groups (`sp_size`).
2. T5: built on CPU in bf16, then FSDP1.
3. CLIP: built in fp16; the visual tower goes to FSDP1 and the text tower is moved to GPU (unused).
4. VAE: weights as stored, bf16 mean/std.
5. DiT: built on `meta` → FSDP2 `fully_shard` per block + root → `to_empty` → rank 0 reads the
   safetensors → `set_model_state_dict(full_state_dict, broadcast_from_rank0, strict=True)`.

`inference_core`, per request:

1. Decord reads the driving video. Frames are resampled to `fps`, resized to `width*height`
   (divisor 16) and zigzag-padded.
2. Loop over clips of `clip_len` frames (1-frame overlap). For each clip:
   * VAE-encode the reference image, the continuation frame and the driving clip.
   * CLIP-visual on the reference image and on the first driving frame.
   * T5 on the prompt, the reference prompt and the negative prompt.
3. `forward_ref` runs **once per clip**. It fills a per-layer K/V cache of the driving-video tokens
   (time embedding fixed at t=1).
4. Each denoising step calls `forward_gen` (cond; plus uncond when CFG > 1, which skips block 9).
   Self-attention is `flex_attention` over `[generated tokens | cached driving K/V]`. It uses the
   frame-aligned In-Context mask (query frame f sees all generated tokens plus driving frame f−1)
   and `score_mod` adds `log_scale` to the keys of generated frame 1 (Base 0.0, Distillation −1.3).
5. `FlowDPMSolverMultistepScheduler` with shift 5. It is used for both models; the YAML
   `flow_solver: euler` is not read.
6. VAE decode, stitch clips, crop the padding, mux the source audio.

## 4. V100 blocker inventory (upstream file:line @ 3ad2fef)

| # | where | what | effect on sm70 | port |
|---|---|---|---|---|
| 1 | `wanxiang/models/attention.py:6-18`, `:98-135` | `flash_attn_interface` (FA3) / `flash_attn` (FA2) varlen kernels | FA2 needs Ampere+ and FA3 needs Hopper, so flash_attn cannot run on Volta. With `FLASH_VER=None` the upstream code hits `assert FLASH_VER == 2` (`:117`). Callers: `wan_animate_2_model.py:196,198` (cross-attn), `:327` (reference self-attn), `clip.py:90,209` (CLIP ViT). | SDPA re-implementation (`attention_sdpa.sdpa_attention`) |
| 2 | `attention.py:22-28`, `:189`; `wan_animate_2_model.py:407` | `torch.compile(flex_attention, mode="max-autotune", fullgraph=True)` with BlockMask + `score_mod` | Triton flex kernels are unsupported on V100. The eager/MATH alternative materialises Q×KV (720P: 79,232 × 154,880 per head). | mask plan + SDPA (`build_mask_plan` / `masked_sdpa_attention`) |
| 3 | `wan_animate_2_model.py:566-611` | `create_block_mask(..., _compile=True)` | inductor/Triton compile at first use | skipped on the SDPA backend; the same `mask_mod` is evaluated in row tiles |
| 4 | `pipelines/wan_animate_2_pipeline.py:301` | global `torch.autocast(dtype=torch.bfloat16)` | bf16 is emulated on sm70: no tensor cores, and SDPA bf16 falls back to MATH (§5) | `prec.dit` (fp16 on V100) |
| 5 | pipeline `:272, :316, :319, :322, :351, :354, :363, :380` | pixel and conditioning tensors `.to(torch.bfloat16)` | same as #4 | `prec.vae` (pixels) / `prec.dit` (conditioning) |
| 6 | pipeline `:631` | `torch.set_default_dtype(torch.bfloat16)` | default-dtype tensors are bf16 | `prec.dit` |
| 7 | pipeline `:649-652` | FSDP2 `MixedPrecisionPolicy(param/reduce/output=bf16)` | bf16 GEMMs; residual stream cast to bf16 between blocks | `param=fp16`, `output=fp32` on V100 |
| 8 | pipeline `:576`, `:581`, yaml `:82-84`, `distributed/fsdp.py:10,38` | T5 bf16, FSDP1 `param_dtype` bf16 | bf16 GEMMs on sm70 | weights stay bf16 as stored; FSDP1 computes in **fp32** (all units resharded after use, §6.3) |
| 9 | pipeline `:301` + `vae.py:779,786` (the VAE's own autocast is commented out), pipeline `:617` | VAE conv3d runs under the bf16 autocast | bf16 conv3d on sm70 has no tensor-core path: slow, and cuDNN support is version-dependent | VAE in **fp32**; mean/std keep upstream's bf16-rounded values |
| 10 | pipeline `:587, :599-601`; `eval_i2v.py:67-75`, `clip.py:632` | CLIP built in fp16, so the checkpoint is rounded **during** `load_state_dict` | fine on V100, but it calls flash_attn (#1) | loaded in fp32, checked, then cast (§6.4); attention via SDPA |
| 11 | demo `:113-116`, yaml `sp_size: 8` | defaults 720×1280, clip_len 81, sized for 8× A800 80 GB | memory must fit in 32 GiB | §7: 480/540P fit comfortably; **720P is tight, so measure first** |
| 12 | pipeline `:751-772` | the launcher spawns `torch.cuda.device_count()` workers itself; it polls only the init events | `torchrun --nproc_per_node=8` would create 64 ranks; a failed worker leaves the parent waiting forever | one `python` process; `main()` refuses `LOCAL_WORLD_SIZE>1`; explicit worker status reporting + liveness monitoring (§6.5) |
| 13 | pipeline `:530, :640`, yaml `:85` | NCCL, device mesh, HYBRID_SHARD | nothing sm80-specific. FSDP2 `reshard_after_forward=True` re-gathers ~28.7 GB of fp16 params per forward per GPU, which costs ~3 s/forward on PCIe-only servers | unchanged (32 GB cannot hold unsharded params) |
| 14 | `requirements.txt:82-88` | torch 2.7.1+cu126, triton 3.3.1 | the cu126 wheels include sm_70; newer CUDA builds may drop Volta | keep cu126/cu118; `probe_v100.py` prints `torch.cuda.get_arch_list()` |
| 15 | `wanxiang/eval_i2v.py:20` | default argument `torch.cuda.current_device()` is evaluated at import | not a V100 issue (it only breaks CPU-only imports) | test-only stub |

No other sm80-only ops were found: no TF32 flags, fp8 is off (`use_fp8_gemm: False`), and there are
no explicit cuDNN-MHA or other Triton kernels. The VAE's own `F.scaled_dot_product_attention`
(`vae.py:298`) runs fp32 on V100, using the mem-efficient or MATH kernel.

## 5. BF16 on V100: what is actually supported

Verified against **torch 2.7.1** (the version upstream pins) and cross-checked against PyTorch
`main`; both agree:

1. `torch.cuda.is_bf16_supported(including_emulation=True)` returns True when `major >= 8`.
   Otherwise, **by default**, it tries to allocate a bf16 tensor and returns True if that works, so
   **on V100 it returns True through emulation**, which is misleading. With
   `including_emulation=False` it returns False on sm70. `is_tf32_supported()` is defined as
   `is_bf16_supported(including_emulation=False)`. Sources:
   [v2.7.1 L177-L209, L212-L221](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/__init__.py#L177-L221)
   and [main ~L244, L276-L288](https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py#L244).
2. Because of (1), `torch.autocast("cuda", dtype=torch.bfloat16)` is **silently accepted** on V100.
   It raises only if `is_bf16_supported()` is False:
   [autocast_mode.py v2.7.1 L323-L331](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/amp/autocast_mode.py#L323-L331).
   CUDA autocast also accepts `dtype=torch.float32` (no dtype whitelist for CUDA), which the model's
   fp32 regions and this port's fp32 fallback rely on.
3. `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` ([v2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)):
   * Flash attention supports sm80–sm120 only (L225-L268; `main`: sm80–sm121).
   * cuDNN MHA supports sm80+ only (L518-L530).
   * Mem-efficient attention runs on sm50+ (L276-L312), but for `major < 8` its allowed dtypes are
     **{Half, Float}** only (L714-L715, L780-L785).
   * So **bf16 SDPA on V100 falls back to the MATH kernel**. That kernel upcasts to fp32 and
     materialises the full [B, H, Lq, Lk] score matrix
     ([attention.cpp v2.7.1 L825-L903](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/attention.cpp#L825-L903)).
     This is why a plain bf16 SDPA swap needs chunking. This port chunks MATH calls to a fixed
     budget (§6.2).
4. Hardware: Volta tensor cores take **FP16** inputs with FP16/FP32 accumulate
   ([V100 whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)).
   BF16 and TF32 tensor-core math arrived with Ampere (A100)
   ([A100 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)).
   FlashAttention-2 targets Ampere, Ada and Hopper ([README](https://github.com/Dao-AILab/flash-attention)).

**Conclusion.** On V100, bf16 works as a *storage format* and through emulated or upcast ops, but it
is **not native**:

* bf16 GEMMs and convolutions cannot use tensor cores;
* bf16 SDPA cannot use any fused kernel;
* Triton refuses bf16 below sm80.

The port therefore computes in **fp16 on tensor cores**. It keeps fp32 where range or precision
matters:

* the residual stream between blocks;
* the model's own fp32 regions (time embedding, modulation, head);
* SDPA softmax and accumulation, which are fp32 inside the kernel;
* T5 and the VAE.

The fp16 range risk is covered by the checks in §6.4. `scripts/probe_v100.py` measures all of this
on the real card.

## 6. Design decisions

### 6.1 Capability switch (A800/H100 unchanged)
* `wanxiang/utils/precision.py:resolve_precision()`. With `auto`:
  * **sm ≥ 80** returns exactly the upstream recipe (bf16 DiT/T5/VAE, fp16 CLIP, guard off, CLIP
    text tower on GPU).
  * **sm < 80** returns fp16 DiT, fp32 residual between blocks, fp32 T5 and VAE, fp16 CLIP, guard
    `first_step`, CLIP text tower on CPU.
* Every V100 knob in `*_v100.yaml` is `auto`, and the upstream FSDP dtype fields are kept (bf16).
  So the V100 YAMLs resolve to **the same recipe as the upstream YAMLs** on sm80/sm90, and to the
  same V100 recipe on sm70. This is tested with mocked capabilities for all four YAMLs.
* Overrides: `--dtype fp16|fp32|bf16|auto` (demo), env `WAN_DTYPE`, YAML `test_cfg.dit_dtype` /
  `t5_dtype` / `vae_dtype` / `clip_dtype` / `overflow_guard` / `clip_text_on_gpu` /
  `fp16_underflow_frac`. `--dtype bf16` resolves to the full upstream recipe with either YAML
  family.
* Attention backend: `test_cfg.attn_backend` or env `WAN_ATTN_BACKEND` = `auto|flash|sdpa`.
  `auto` means upstream kernels on sm ≥ 80 when flash_attn is installed, and SDPA otherwise.
  Upstream would have crashed in the second case.
* On sm ≥ 80 with flash-attn and the upstream recipe, **the original compute recipe and tensor
  operations are preserved**. Every port-only tensor operation is gated to the port path (SDPA
  backend or non-bf16 precision):
  * K/V clones;
  * K/V and VAE cache frees;
  * pixel frees;
  * the T5 fp32 autocast wrapper;
  * CLIP fp32 load and check;
  * the guard and the conditioning checks.

  A test checks that the flash path caches upstream's K/V views (no copy).
* **Applied everywhere** (control plane, no tensor work):
  * the launcher protocol (worker status reports, per-request completion acks, failure latching,
    liveness checks, `close()`, §6.5);
  * the `LOCAL_WORLD_SIZE` launch guard;
  * the T5 fp16 refusal;
  * small Python-side checks (backend dispatch per attention call, guard-off wrapper calls).

### 6.2 Attention (`wanxiang/models/attention_sdpa.py`, dispatched from `attention.py`)
* **flash_attention → `sdpa_attention`.**
  * Same layout `[B, L, N, D]`.
  * Like flash varlen packing, sample b only ever sees `k[b, :k_lens[b]]`. Padding is **sliced
    away**: per sample, or grouped by equal length. It is never just masked, because a masked
    NaN/inf padding value would still poison MATH (0 × NaN).
  * `k_lens` bounds are validated.
  * `softmax_scale` and `q_scale` are supported. `causal` and windowed attention raise (unused by
    this model).
  * fp32 q/k (after RoPE) are cast **once** to the compute dtype; upstream went fp32 → bf16 →
    v.dtype.
  * `q_lens` is not used by the model. Here rows ≥ `q_lens[b]` return 0; upstream's wrapper would
    fail in `unflatten` on genuinely shortened query totals.
* **flex_attention + BlockMask + score_mod → `build_mask_plan` + `masked_sdpa_attention`.**
  * The plan evaluates the *same* `mask_mod`, which was moved verbatim from `create_mask` into
    `incontext_mask_mod`. It works in row tiles of ~16 M elements, so the dense 12.3 G-element
    (720P) mask is never formed.
  * Consecutive identical rows are grouped into runs. Each query frame is one run, and the padding
    rows form a fully-masked run whose output is 0, as in flex.
  * Each run stores only its visible key columns, about q_limit + hw: 82,800 keys instead of
    154,880 at 720P.
  * `score_mod` is converted into an additive bias once per (plan, score_mod). It is verified to be
    additive and head-independent and rejected otherwise. For Base (`log_scale=0`) the bias is
    all-zero, so it is dropped and SDPA runs without a mask.
  * The plan is cached per resolution in `model.block_masks`, like upstream's BlockMask.
* **Kernel selection.** Every caller backend flag is honoured, including MATH.
  * Each SDPA call first tries the fused kernels (flash / mem-efficient / cuDNN) that the caller
    still enables. On V100 that means mem-efficient for fp16/fp32, with `WAN_SDPA_Q_CHUNK` (4096)
    query rows per call.
  * If none of them accepts the inputs (e.g. bf16 on sm70) **and MATH is enabled**, it uses MATH in
    query chunks sized so the fp32 scores and softmax stay within `WAN_SDPA_MATH_BUDGET_MB`
    (1024 MiB). At 720P that is 259 rows per call. A WARNING is logged once per (device, dtype,
    mask dtype, dropout).
  * If MATH is disabled, PyTorch's error is re-raised.
  * There is **no availability cache**. A rejection costs one dispatch-time exception and never
    affects later calls with different inputs, e.g. a dropout call followed by a no-dropout call.
* **Exact semantics kept, including upstream quirks.**
  * Zero-padded tokens inside `[0, q_limit)` (frames beyond the clip, and positions when the latent
    grid is smaller than `width*height/256`) are *not* masked and still absorb softmax mass.
  * Query rows that upstream discards are still computed: about 18% of attention FLOPs for a
    65-frame input with `clip_len=81`.
  * The bias is stored in the query dtype, as SDPA requires. In fp16, −1.3 becomes
    −1.2998046875, a 1.95e-4 logit error.

### 6.3 Precision and memory
* **DiT.** Storage stays the upstream fp32 FSDP2 shards (7.6 GiB/GPU), and the bf16 checkpoint
  upcasts exactly. FSDP2 casts each block to fp16 when it gathers it. The residual stream between
  blocks stays **fp32** (`output_dtype=fp32`). Upstream bf16 never overflows there; fp16 could.
* **Reference K/V cache.** With context parallelism, the cached K/V used to be `chunk()` views of
  the all_to_all `[q|k|v]` buffer, which kept Q alive too (+50%, 3.1 GB/GPU at 720P). They are now
  cloned into their own storage on the SDPA (port) path; the flash path keeps upstream's views. Values are unchanged.
* **T5.** fp32 compute.
  * **`t5_dtype=fp16` is refused** at start-up on every path, including `param_dtype=float16` on
    the upstream recipe. umT5/T5-v1.1 is widely reported to overflow in fp16, and its `fp16_clamp`
    turns inf into finite ±64512 values. Round 2 reproduced a finite wrong embedding with
    fp16-castable weights, so only fp32 or bf16 are allowed.
  * T5 runs on 2–3 prompts per clip, about 14 TFLOP per GPU in fp32. **Hypothesis, not measured:**
    a few seconds per clip, well under 1% of a Base clip's DiT FLOPs.
  * The weights stay **bf16 as stored**, sharded 8-way (1.3 GiB/GPU), and FSDP1 computes in fp32.
  * FSDP1 deliberately keeps the **root** unit's unsharded params after forward (torch 2.7.1
    `_runtime_utils.py:482-496`). For umT5 that root holds the token embedding plus the final
    norm: 1.05 B params, 4.2 GB in fp32.
  * On the V100 path these two modules are wrapped as their own FSDP units (`extra_wrap`), so every
    unit is resharded after encoding. Nothing stays gathered: 0 bytes vs 4.2 GB.
  * The transient while encoding is the fp32 embedding gather (4.4 GiB).
  * Outputs are bit-exact vs an fp32 reference (2-rank test).
* **VAE.** fp32. `VideoVAE`'s own default is `dtype=torch.float` (`vae.py:718`); upstream only runs
  it in bf16 through the pipeline-wide autocast. fp16 VAEs are a known overflow risk.
  * The causal-conv caches (`_feat_map` / `_enc_feat_map`) used to stay alive until the next call:
    several GiB at 720P during denoising. They are now released after every encode and decode.
  * The fp32 pixel tensors are freed before denoising.
  * **No VAE tiling** exists in this code base. The traced fp32 decode peak is **~16 GiB at 720P**,
    and that is the 720P limiter (§7).
* **CLIP.** Loaded in fp32 (so checkpoint values are not rounded during `load_state_dict`), checked,
  then cast to fp16 as upstream. The cost is +2.4 GB host RAM per worker during init.
  * The text tower stays on CPU (`clip_text_on_gpu: auto` → false for an fp16 DiT; saves
    ~1.1 GB/GPU). It is never used at inference.
  * Fallback if the check fails: `clip_dtype=fp32`.

### 6.4 "Load dict must not break", and the fp16 checks
* **No parameter keys were renamed, added or removed.** The model diff:
  * moves `create_mask`'s closure out to a module function;
  * adds an `if`;
  * clones the cached K/V.

  The tests prove the key, shape and dtype set equals upstream (tiny config) and equals the **real
  release header** (1303 tensors, full config on the `meta` device, all four YAMLs).
* The **DiT loader path is unchanged**. `build_dit()` holds upstream's statements (meta →
  fully_shard → `to_empty` → rank-0 `load_safetensors` →
  `set_model_state_dict(strict=True, broadcast_from_rank0)`), moved into a function so it can be
  tested. It still fails loudly on an extra or missing key: torch 2.7.1 raises a `KeyError` or
  "Missing key(s)" naming the tensor.
* **Every dtype cast happens after a strict load:**
  * DiT: FSDP2 casts the loaded fp32 shards per forward;
  * T5: FSDP1 MixedPrecision;
  * CLIP: `.to(fp16)` after the fp32 load and check;
  * VAE: `.to(fp32)`.

  None of these can touch key names.
* **Load-time fp16 check** (`fp16_guard.check_fp16_castable`, DiT and CLIP visual). For the DiT it
  runs on the GPU shards and is all-reduced over the shard group. Per module:
  * **non-finite** → always FAIL, even when allow-listed;
  * **original |w| > 65504** → FAIL. This tests the source value, so 65504 < |w| < 65520, which
    would round down to a finite 65504, also fails;
  * **underflow** (nonzero |w| ≤ 2^-25 ≈ 2.98e-8 becomes 0) → FAIL if more than
    `fp16_underflow_frac` of the module's nonzero weights (default 0.001; 0 = strict). **Every**
    flushed weight is logged as a WARNING: per-module counts, max flushed |w| and a total. The
    error bound and the (unmeasured) rationale are in §13 F4.
  * fp16 subnormals (|w| < 2^-14 ≈ 6.10e-5) → reported only.

  The check **raises** listing the modules, unless they are in the explicit DiT allow-list
  `test_cfg.fp32_modules`. Allow-listed modules get their own FSDP2 unit with fp32 params and run
  fully in fp32. For CLIP, the fix hint points to `clip_dtype=fp32`.
* **Server pre-flight.** `scripts/check_ckpt_fp16_range.py` streams the checkpoints tensor by tensor
  (`safe_open`, or `torch.load(mmap=True)` for `.pth`). It uses the **same functions and
  thresholds**, prints max |w|, the flush list and subnormal counts, then PASS/FAIL and a suggested
  `fp32_modules: [...]`.
* **Runtime checks** (`fp16_guard.OverflowGuard` + `attention_sdpa.MONITOR`). These are on whenever
  the guard is not `off`, which is the default for an fp16 DiT.
  * **Attention kernel monitor, every forward, every step.** Inside `sdpa_attention` and
    `masked_sdpa_attention`, q/k/v (after the dtype cast, after RoPE, after `k_lens` slicing) and
    the output of every call update a device-side inf/NaN flag (`aminmax`, no host sync).
  * This closes the gap the review found (§13 F1). An overflowed q can produce an all-`-inf` logit
    row, which SDPA turns into a **finite** wrong output (zeros). The output hooks and the model
    sentinel could not see it.
  * **Model-output sentinel**, every forward.
  * **Detailed hooks**: `first_step` covers the reference pass and step 0 (cond and uncond) of every
    clip, plus any forward whose flag tripped, which is re-run with hooks. `all` covers every step.
    Hooks record max|x| of:
    * block outputs;
    * `pre_self_attention` (q/k/v + the six **modulation** chunks);
    * the raw attention output;
    * cross-attention;
    * FFN up-projection and output;
    * the root modules.

    In this mode the kernel monitor also records each kernel tensor with a label, in execution
    order. That gives the first offender, e.g. `blocks.3.block.cross_attn:attn1.q`.
  * **Collectives.**
    * A non-detailed forward does **1 all-reduce** of 2 floats (sentinel + kernel flag) and 1
      device→host readback.
    * A detailed forward does **2 all-reduces**: record counts as [max, −max] first, so a count
      mismatch makes *every* rank raise the same error, then all flags and values. It also does
      2 readbacks.
    * Whether a forward is detailed is identical on all ranks (step schedule plus collective rerun
      decisions), and only detailed forwards produce records. So skipping the count reduction on
      non-detailed forwards keeps the agreement guarantee.
    * Records are cleared before every attempt.
    * The conditioning checks add 1 all-reduce + 1 readback per group.
    * Nothing raises inside a hook or kernel.
  * Thresholds: **ERROR** is any inf/NaN. **WARNING** is max|x| ≥ `overflow_warn_at` (default
    16384 = 2^14).
  * On ERROR it logs the first offending module and the step/branch, then:
    * `overflow_action: abort` raises `FP16OverflowError` with a fix hint;
    * `overflow_action: fp32` (default) switches that **whole block** to fp32 activations (args
      upcast, fp32 autocast that also upcasts the fp16-gathered weights, fp32 SDPA). It re-runs the
      same forward, so the scheduler state is untouched, and logs `FALLBACK`.
  * **Conditioning checks.** After the VAE encodes, both CLIP calls (output + CLIP's own attention
    kernels), the T5 calls, and the VAE decode (**before** the [-1, 1] clamp, which would turn inf
    into ±1), one collective check runs per group. All ranks raise the same `FP16OverflowError`
    naming the tensor and the knob to change.
  * **Not checked:** finite-but-degraded results that never reach inf/NaN. That covers fp16 rounding
    drift and large finite values that get normalised away. WARNINGs flag values near the limit; an
    fp32 run (`--dtype fp32`) is the reference for quality.
  * Overhead: 4 `aminmax` per attention call, ~15 more per block on detailed steps, plus the
    collectives above. **Not measured on GPU.**

### 6.5 Launcher failure handling (§13 F2, N1)
Upstream's `main()` polled only the per-worker init events. A worker that failed (now including a
rejected checkpoint) never set its event, so the parent waited forever. Round 2 also showed that a
failure report could be discarded if rank 0's result arrived next.

* **Worker status queue.** Workers report `init_ok` / `init_fail` / `fail` (with the original
  error). After **every** completed request each worker also sends `('done', gpu, request_id)`.
* **`WorkerMonitor` (parent)**:
  * **latches** every report: once a failure was seen, nothing can succeed any more;
  * checks worker liveness, allowing a short grace period for a dying worker's report;
  * keeps rank 0's result separate from worker status.
* **Accepting a request.** It succeeds only when all of these hold:
  * the result has arrived;
  * **every** worker has acked that `request_id`;
  * a final latch and liveness check passes.
* **On any failure or dead worker**, it terminates and joins all workers and raises
  `RuntimeError("Wan-Animate-2 <initialisation|inference> failed: worker N: <original error> …")`.
  The demo exits non-zero. `WAN_INIT_TIMEOUT_SEC` and `WAN_REQUEST_TIMEOUT_SEC` add optional hard
  timeouts.
* A failed worker does not enter the final barrier, because its peers may be in another phase.
* `pipeline.close()` stops the non-daemon workers; upstream's demo would otherwise hang at
  interpreter exit. The demo calls it in a `finally`.
* **Residual window.** A worker that dies *after* every ack arrived and the final liveness check
  passed is reported by the next request (or by `close()`), not the finished one.

## 7. Memory and time (`scripts/estimate_v100_memory.py`)

* **Persistent buffers** are counted analytically from the real parameter counts.
* **Transient peaks are traced.** The real VAE encode/decode and one full-size DiT block
  (`forward_ref`, `forward_gen`, per-rank shapes, CP all_to_all buffers produced by `ops.all_to_all`
  itself) run under FakeTensorMode, and every allocation is followed until it is freed.
* **Not traced:** fused-kernel workspaces, the CUDA context, NCCL buffers and fragmentation. A flat
  **1.5 GiB** is added for these.
* Units are **GiB per GPU vs 31.75 GiB** (nvidia-smi total of a V100 32GB). OK < 80%, TIGHT < 100%.

Example: a 64-frame 25 fps driving video (→ 61 at 24 fps → padded to 65, one clip), `clip_len` 81,
8 GPUs, sp=8:

| GiB per GPU (component rows: without overhead; **phase rows: including the 1.5 GiB overhead**) | 480P | 540P | 720P |
|---|---|---|---|
| persistent: DiT fp32 shards 7.63 + FSDP2 root 0.45 + T5 1.32 + CLIP 0.15 + VAE 0.47 | 10.0 | 10.0 | 10.0 |
| reference K/V cache (owned, fp16) | 2.53 | 3.31 | 5.84 |
| traced VAE encode / decode transient | 5.5 / 7.0 | 7.2 / 9.1 | 12.8 / 16.1 |
| traced one-block `forward_gen` transient (fused / MATH / fp32 block) | 2.0 / 2.3 / 3.1 | 2.4 / 2.9 / 3.6 | 3.7 / 4.2 / 5.2 |
| **1 conditioning** (VAE encode, T5 4.4 GiB embedding gather, CLIP) | 17.7 | 19.5 | **25.6 TIGHT** |
| **2 reference pass** | 17.0 | 18.0 | 21.2 |
| **3 denoising** fused / MATH fallback / a block in fp32 | 17.7 / 17.9 / 19.0 | 18.8 / 19.3 / 20.3 | 22.7 / 23.2 / 24.5 |
| **4 VAE decode** (K/V freed) | 18.6 | 20.7 | **27.7 TIGHT** |
| same, full 81-frame clip: peak phase | 19.9 | 21.4 | **27.9 TIGHT** (fp32-block denoising 26.5) |

**Verdict:** 480P and 540P fit with a margin of 11+ GiB. **720P is conditional on measurement.**

* fp32 VAE decode and encode take it to 26–28 GiB of 31.75.
* Fragmentation of the many large VAE activations is not modelled.

If 720P OOMs in the VAE, two options remain:

* `vae_dtype=fp16` halves the VAE memory but is an unguarded overflow risk; the decode check would
  at least detect inf/NaN.
* Spatial tiling (diffusers' `AutoencoderKLWan` has it) would need porting.

The previous formula-only estimate ("720P fits, ~24 GB") missed ~7.3 GB/GPU: the Q kept alive by
the K/V cache views and the retained T5 root. Both are now eliminated rather than just counted.

**Time** (FLOP-based guess: 40 TFLOPS attention, 80 TFLOPS GEMM; NVLink ~100 GB/s or PCIe
~10 GB/s; excludes load, T5, VAE):

| | 480P | 540P | 720P |
|---|---|---|---|
| s per forward, NVLink / PCIe | ~4.6 / ~7.8 | ~7.2 / ~10.5 | ~19.7 / ~23.6 |
| Base, 80 forwards per clip | ~6 / ~10 min | ~10 / ~14 min | ~26 / ~31 min |
| Distillation, 10 forwards per clip | ~0.8 / ~1.3 min | ~1.2 / ~1.8 min | ~3.3 / ~3.9 min |

**Host RAM** (estimate, not measured): peak **~180–210 GB** during initialisation.

* Each of the 8 workers holds the bf16 umT5 module plus its `torch.load`ed state dict (~22.7 GB)
  at the same time; this is upstream behaviour. Add the per-process overhead.
* The port loads CLIP in fp32 (model 4.8 GB + fp32 state dict 4.8 GB ≈ 9.5 GB per worker, against
  7.2 GB upstream) so it can check it before the fp16 cast. This happens after T5 has moved to the
  GPU, so it does not raise the peak.
* Rank 0's 33 GB DiT state dict is loaded later still, when the other workers' CPU footprint is
  small.

**Recommended order:**
1. Distillation at 480P.
2. Base at 480P or 540P.
3. Compare `torch.cuda.max_memory_allocated` with this table.
4. Only then try 720P.

## 8. Test results (CPU only, `.venv-cpu`, torch 2.7.1)

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../.venv-cpu/bin/python -m pytest tests -q` → **98 passed**
(~100 s). All modified and new `.py` files pass `py_compile`. Round 2 added 16 tests; they are
listed in §13 (round 2).

* `test_v100_attention.py` (39 tests):
  * SDPA vs a float64 dense softmax reference with flash-varlen `k_lens` semantics (equal and
    unequal lengths, query chunking 4096 vs 7) in fp32 (atol 2e-5) and fp16 (3e-3); scale /
    q_scale / q_lens; fp32→half cast rule.
  * `incontext_mask_mod` == upstream's own closure (captured from the unmodified upstream file
    loaded from git) == an independently written frame-aligned mask, for 4 geometries. The plan
    reconstructs the dense mask exactly.
  * `masked_sdpa_attention` == upstream flex_attention (PyTorch's flex reference implementation,
    BlockMask from the same `mask_mod`, upstream's own `score_mod`) == the dense float64 reference,
    for log_scale 0/−1.3, fp32/fp16 and 3 geometries. Padding rows are 0.
  * General masks and q-dependent biases work; non-additive and head-dependent score_mods are
    rejected.
  * **Whole tiny model**: the patched SDPA model equals the unmodified upstream model with
    reference kernels (atol 1e-5).
  * **New:**
    * NaN in discarded padding never reaches the output (both the auto and MATH backends);
    * `k_lens` bounds are validated;
    * the fused→MATH fallback produces budget-sized chunks with exact results;
    * an outer `sdpa_kernel` context is respected.
* `test_v100_state_dict.py` (15 tests):
  * The key/shape/dtype set equals upstream (tiny) and the real release header (all four YAMLs).
  * The pipeline's own `build_dit` + `load_safetensors` on 2 gloo ranks is bit-exact; the sharded
    forward equals the unsharded forward; an extra or missing key fails loudly, naming the key.
  * **New:**
    * T5 under FSDP1 with `extra_wrap` (2 ranks): bf16 storage, fp32 output bit-exact (< 1e-5) vs
      fp32, **0 bytes retained in every unit**;
    * a **context-parallel sp=2 forward** of the tiny model on 2 gloo ranks (all_to_all emulated
      via all_gather in the test) equals the single-process forward, and the cached K/V own their
      storage (the test fails without the clone);
    * all four YAMLs resolve identically per family under mocked sm80/sm90/sm70, equal to the
      upstream recipe on sm ≥ 80;
    * `--dtype bf16` with a V100 YAML gives the upstream recipe.
* `test_v100_fp16_guard.py` (24 tests):
  * Exact thresholds: 65519/65520/1e6 are out of range; 2.9e-8 flushes to 0; 3e-8 and 1e-5 are
    subnormal.
  * Overflow, NaN and underflow FAIL; the allow-list exempts explicitly.
  * The underflow tolerance logs a WARNING with count and max |w|; `underflow_frac=0` fails on one
    flush.
  * The 2-rank loader check catches a value placed only in rank 1's shard.
  * Hook coverage; a real CPU-fp16 FFN overflow is caught and the fp32 fallback recovers.
  * Near-limit warnings; sentinel re-localisation; 2 ranks agree on a one-rank NaN; scanner
    PASS/FAIL/allow-list.
  * **New:**
    * The review's concealed cross-attention overflow is caught in `first_step` (non-detailed step 3,
      via the kernel flag) and in `all`, naming `blocks.0.block.cross_attn:attn0.q`. With
      `action=fp32` the block is recomputed in fp32 and gives the fp32 answer (ones, not zeros);
    * kernel labels follow execution order;
    * a 2-rank record-count mismatch makes **both** ranks raise the same error;
    * `assert_finite` is collective (rank-1-only NaN → both ranks raise) and sees the CLIP
      attention flag and inf;
    * CLIP is checked **before** the fp16 cast (a 1e-9 weight is seen; the hint is
      `clip_dtype=fp32`).
* `test_v100_launcher.py` (4 tests, **new**). These use the real `WanAnimate2MPIPipeline` → `main`
  → `mp.spawn` → `worker` → `build_dit` path on CPU/gloo, with GPU-only calls and T5/CLIP/VAE
  stubbed, in a subprocess with a 180 s timeout:
  * a rejected checkpoint exits non-zero with the original `FP16OverflowError`;
  * a clean run returns the result and exits;
  * a worker that dies or raises during inference makes the call raise and the process exit
    non-zero instead of hanging.
* **Mutation checks:**
  * dropping the score_mod bias → 7 attention tests fail;
  * dropping the reference-frame keys → 14 fail;
  * removing the K/V clone → the CP storage test fails.
* **The review repro scripts** were re-run unchanged from `review/`, with CWD in a scratch directory
  so nothing was written to `review/`:
  * `repro_review.py` now stops at its first section with
    `FP16OverflowError … first offending module blocks.0.block.cross_attn:attn0.q`.
  * A copy with only per-section try/except added shows:
    * VARLEN gives ones (= the sliced reference);
    * UNDERFLOW (1 of 2000) is accepted within tolerance with a WARNING;
    * OVERMAX (65519) is rejected;
    * CACHE_BYTES is unchanged, because it is a synthetic `chunk()` demo that does not call the
      model.
  * `repro_guard_count.py`: both ranks raise `different tensor counts (min 1, max 2 …)`, no
    timeout.
  * `repro_fsdp_memory.py` still shows retention, because it wraps T5 without the new
    `extra_wrap`. A copy adding the pipeline's `extra_wrap` shows 0 bytes retained in every unit.
* Probe: `scripts/probe_v100.py` exits cleanly without CUDA. `--cpu-smoke` exercises the
  table/except paths.

## 9. Known differences vs the official A800 path
1. fp16 instead of bf16 compute. fp16 has 3 more mantissa bits and a much smaller range. Outputs
   will not be bit-identical; expect small numeric drift, not a different model. The residual
   stream is fp32 here (bf16 upstream).
2. T5 and VAE run in fp32 (bf16 upstream), which is more precise. The VAE normalisation constants
   keep upstream's bf16-rounded values on purpose.
3. SDPA kernels instead of FlashAttention/flex. The maths is identical up to float rounding. The
   distillation `log_scale` bias is fp16-rounded (1.95e-4 logit).
4. Pixel tensors are fp32 instead of bf16 (more precise VAE inputs).
5. With the guard in `fp32` mode, a block that overflowed runs fp32 activations from then on. It is
   logged as ERROR `FALLBACK`.

## 10. Open risks
* **fp16 overflow in activations.** This is unknown for these checkpoints (the distilled model
  especially). It is now detected at the attention kernel boundary and the model output on every
  step, and in conditioning tensors. Remaining blind spot: finite-but-degraded values (§6.4). If
  many blocks overflow, speed drops toward fp32 (no tensor cores, ~8× slower for those blocks).
* **720P memory**: TIGHT (26–28 GiB of 31.75) because of fp32 VAE encode/decode. Allocator
  fragmentation is unmodelled. **Measure first.**
* **Speed.** The estimates assume 40 TFLOPS fp16 attention and 80 TFLOPS GEMMs on V100; real
  numbers may be lower (`probe_v100.py` measures them). The PCIe-only topology adds ~3–4 s per
  forward.
* **Kernel selection.** If mem-efficient SDPA rejects the fp16 + broadcast-bias case on the real
  card, the code falls back to budget-chunked MATH. That is correct and memory-bounded, but much
  slower. It logs a WARNING.
* **Host RAM** (~180–210 GB estimated at init, not measured; §7).
* **Launcher residual window:** a worker death after all acks and the final liveness check is
  reported by the next request, not the finished one (§6.5).
* **Untested GPU-only code:**
  * `force_fp32` on FSDP2 units under CUDA autocast(fp32);
  * NCCL versions of the collective checks;
  * `destroy_process_group` after a failure. The parent kills workers that hang there.
* The CPU tests exercise gloo, FSDP1/FSDP2 and a 2-rank context-parallel forward, not NCCL with 8
  ranks.

## 11. What still needs a real V100 run (in order)
1. `python scripts/probe_v100.py --json probe.json`. Confirm:
   * sm70 is in `torch_arch_list`;
   * `is_bf16_supported(including_emulation=False)` is False;
   * fp16 vs bf16 matmul and conv3d TFLOPS;
   * SDPA EFFICIENT runs fp16 with the bias (720P chunk shape) and bf16 falls to MATH.
2. `python scripts/check_ckpt_fp16_range.py ckpts/wan_animate_2/*.safetensors --json fp16_range.json`.
   Expect PASS. If it FAILs, copy the suggested `fp32_modules` into the V100 YAML. Read the flush
   WARNING list either way.
3. `RES=480p scripts/run_v100_distill.sh` (defaults to `examples/demo1`). Check the following:
   * the log line `precision: dit=fp16 … guard=first_step …` and `attention backend: sdpa`;
   * the `[fp16-check]` summaries;
   * any `[fp16-guard]` WARNING/ERROR;
   * any `[sdpa] no fused SDPA kernel` WARNING (should be absent for fp16);
   * peak memory (`nvidia-smi`, and the per-phase table in §7);
   * seconds/step from tqdm;
   * whether the output video looks right.
4. `RES=480p scripts/run_v100_base.sh`, then 540P, then 720P only if the measured peaks track §7.
5. Quality A/B on the same seed, prompt and inputs:
   * `--dtype fp32` on V100 (slow reference) vs fp16. A difference beyond small drift means a
     precision problem the inf/NaN checks cannot see.
   * Ideally an A800 output, or the ModelScope demo.
6. Measure the guard overhead (`overflow_guard: off` vs `first_step` vs `all`) and tune
   `WAN_SDPA_Q_CHUNK`.

## 12. Diffusers (PR #14412) as an alternative
[huggingface/diffusers#14412](https://github.com/huggingface/diffusers/pull/14412) ("support-wan-animate-2")
was **still open and unmerged** on 2026-10-13. Its `transformer_wan_animate_2.py` vendors the same
`flash_attn_varlen_func` plus `torch.compile(flex_attention, mode="max-autotune")` and
`create_block_mask` code, so it has the **same V100 blockers**. It is not a shortcut. If it is ever
preferred, `attention_sdpa.py` ports over unchanged: same function signatures, same mask_mod and
score_mod. Diffusers' `AutoencoderKLWan` tiling would help VAE memory.

## 13. Review responses (independent review, findings F1–F8)

| # | severity | finding | decision | what was done | evidence |
|---|---|---|---|---|---|
| F1 | blocker | an fp16 overflow inside attention can produce a finite wrong output (an all-`-inf` logit row → SDPA returns 0) that no guard saw; CLIP was unguarded | accepted | `attention_sdpa.MONITOR`: a device-side inf/NaN flag over q/k/v/out of **every** SDPA call (post-cast, post-RoPE, post-slicing), every step. The guard reads it in its single per-forward collective; labelled records in detailed mode. Collective `assert_finite` on CLIP (incl. its kernels), T5, VAE latents and decoded frames (before the clamp). The README no longer says results "can never be silently wrong"; it lists what is and is not checked. | `test_concealed_attention_overflow_is_caught` (first_step non-detailed step, all, step 0; abort + fp32 recovery), `test_monitor_labels_…`, `test_assert_finite_*`; the review's `repro_review.py` now raises |
| F2 | major | a rejected checkpoint leaves `main()` waiting forever | accepted | worker status queue (init_ok / init_fail / fail with the original error); `main()` and `__call__` watch reports **and** process liveness, terminate/join peers, raise; failed workers skip the final barrier; `close()`; the demo exits non-zero | `test_v100_launcher.py` (4 tests via the real launcher path, 180 s bound) |
| F3 | major | the memory estimate missed ~7.3 GB/GPU (Q kept alive by K/V views; FSDP1 retains the 4.2 GB T5 root) | accepted | (a) cached K/V cloned (owned storage); (b) T5 embedding + final norm as their own FSDP1 units: 0 bytes retained, fp32 compute exact, bf16 storage; (c) new per-phase estimator with FakeTensor-traced VAE and DiT-block transients, MATH and fp32-fallback modes, GiB vs 31.75; the traces also exposed retained VAE caches (now freed) and the ~16 GiB fp32 VAE decode peak; 720P marked conditional | CP storage test (+ mutation), T5 retention test, §7 table |
| F4 | major | load checks weaker than required (overflow only when the cast gives inf; underflow tolerated) | **partially accepted** | (a) overflow now tests the **original** \|w\| > 65504 (load check and scanner). (b) The per-module underflow tolerance is **kept**, but made loud and configurable: a WARNING lists every module with any flushed weight (count, fraction, max flushed \|w\|) plus a total; `test_cfg.fp16_underflow_frac` (0 = strict). Error bound: a flushed weight has \|w\| ≤ 2^-25 ≈ 3e-8, an absolute error ~300× smaller than ordinary fp16 rounding of a typical 0.02 weight (2^-11 × 0.02 ≈ 1e-5). Per module, at most 0.1% of nonzero weights are affected, each by ≤ 3e-8. *Hypothesis (no full checkpoint scan or quality ablation yet):* a zero tolerance would force many Linears into fp32 without a visible quality gain. The server scan prints the real flush list. | `test_fp16_cast_thresholds_exact`, `test_load_check_original_magnitude_overflow_fails`, `test_load_check_underflow_tolerance_is_loud_and_configurable` |
| F5 | major | the CLIP check ran after an irreversible fp16 cast; the hint pointed to the DiT allow-list | accepted | CLIP is strictly loaded in fp32 (same keys), checked, then cast; the hint is `clip_dtype=fp32` | `test_clip_checked_before_the_fp16_cast` |
| F6 | major | a record-count mismatch sent ranks into different branches (deadlock) | accepted | counts reduced as max and −min in one collective before the packed reduction; every rank raises the same error | `test_count_mismatch_aborts_on_every_rank`; `repro_guard_count.py` now: both ranks raise |
| F7 | minor | the V100 YAMLs did not restore the full upstream recipe on sm80 (T5 fp32, guard on) | accepted | upstream FSDP dtype fields restored; every V100 knob is `auto` and resolves to upstream values when the DiT runs bf16 | `test_v100_yaml_auto_equals_upstream_yaml` (sm80/sm90/sm70 × 4 YAMLs), `test_dtype_bf16_forces_upstream_recipe_with_v100_yaml` |
| F8 | minor | unequal `k_lens` used masking, so padding NaN contaminated the output; the q_lens wording was wrong | accepted | per-length slicing (grouped by length), bounds validation, q_lens described accurately | `test_unequal_k_lens_slice_padding_away`, `test_k_lens_bounds_are_validated` |
| extra | — | publication hygiene | done | README/NOTES name no person; scripts default to the upstream `examples/demo1` (reference.png, template.mp4, prompt.txt) and require `PROMPT` when `REF_IMG` changes; no private paths; the custom caption appears only as a commented "your own inputs" example | `scripts/run_v100.sh` dry-runs |

### Round 2 (N1–N5, R4)

The round-2 review confirmed F3–F8 as fixed and F1/F2 as partial (via N2/N1). Its repros were copied
to the scratchpad and run there; `review/` was not touched.

| # | severity | finding | decision | what was done | evidence |
|---|---|---|---|---|---|
| N1 | major | the launcher could discard a received worker failure and return success if rank 0's result came next (r2 `launcher_race.py`: exit 0) | accepted | `WorkerMonitor` latches every report across all drains; liveness is checked before accepting; per-request `done` acks from **all** workers; result kept separate from status; final latch + liveness check right before returning (§6.5) | `test_worker_failure_during_inference_raises[fail_before_result \| result_before_peer_failure \| peer_death_at_completion]` via the real launcher path (180 s bound); a mutation that accepts the result without acks fails 2 of the 3; the review's `launcher_race.py` copy now exits 1 |
| N2 | major | `t5_dtype=fp16` was accepted; T5's own `fp16_clamp` turns real fp16 infinities into a finite wrong embedding that passed the checks | **rejected (option removed)** | `resolve_precision` raises a clear error for fp16 T5 on every path (`t5_dtype`, or `param_dtype=float16` on the upstream recipe); only fp32 (V100 default) or bf16 | `test_fp16_t5_is_refused_on_every_path` (4 configs), `test_fp32_and_bf16_t5_are_allowed`; the review's `repro_t5_sharded.py` copy now stops on both ranks with that ValueError |
| N3 | minor | SDPA helper ignored a disabled MATH backend; a negative availability cache keyed on (device, dtype, mask dtype) poisoned unrelated calls | accepted | every caller flag honoured, MATH used only if enabled, otherwise PyTorch's error re-raised; availability cache removed (cost: one dispatch-time exception per rejected call) | `test_disabled_math_is_not_used_behind_the_callers_back[[] \| efficient_only]`, `test_backend_choice_is_per_call_not_cached`; the review's `repro_backend_cache.py`/`repro_new.py` copies (see below) |
| N4 | minor | docs claimed one collective/host sync per forward; there were two all-reduces and extra scalar readbacks | accepted, and reduced | non-detailed forwards: 1 all-reduce + 1 readback (the count reduction is skipped where no records can exist); detailed: 2 all-reduces + 2 readbacks (count readback merged into one transfer); docs state these numbers | `test_guard_collectives_per_forward` (step 3: 1, step 0: 2); the count-mismatch 2-rank test still passes; the review's `repro_guard_accounting.py` copy now shows 1 reduction |
| N5 | minor | doc claims beyond the evidence (nothing changes on A800; two host-RAM ranges; "costs nothing measurable"; "no measurable accuracy gain") | accepted | README: the original compute recipe and tensor operations are preserved on sm≥80, and launcher/error-handling changes apply everywhere; one host-RAM estimate (~180–210 GB, unmeasured) with the fp32-CLIP explanation; T5 cost and the underflow-tolerance rationale labelled as hypotheses (the 3e-8 bound kept) | README / NOTES §6.1, §6.3, §7, §13 F4 |
| R4 | — | the A800 path still performed K/V clones and cache clearing | accepted | K/V clones only on the SDPA backend; K/V clearing, VAE cache clearing, pixel frees and the T5 autocast wrapper only on the port path; launcher changes kept and disclosed | `test_kv_clone_only_on_the_port_path` (flash path: upstream views, SDPA path: own storage) |

**Round-2 repros.** Every file in `review/r2_artifacts/` was copied into the scratchpad and run from
there with `PYTHONPATH=repo:repo/tests`; `review/` was hash-checked unchanged.

| repro (copy) | result now |
|---|---|
| `launcher_race.py` (`WAN_TEST_SCENARIO=late_success`) | **exit 1**: `RuntimeError: Wan-Animate-2 inference failed: worker 1: ValueError: injected failure before rank-0 result` (previously exit 0 with `RESULT stub-ok`) |
| `repro_t5_sharded.py` | refused on both ranks at `resolve_precision`: `ValueError: T5 in fp16 is not supported …` (previously a finite wrong embedding was accepted) |
| `repro_backend_cache.py` | fails as written (`_FUSED_OK` no longer exists by design). A copy with only the `_FUSED_OK` lines removed gives `DROPOUT_CALL ['FUSED','MATH']`, then **`NEXT_NORMAL_CALL ['FUSED']`** (previously `['MATH']`) |
| `repro_new.py` | Its T5 part builds an fp16 T5 by hand, bypassing the pipeline, so it still shows the clamp hiding the overflow. That is why fp16 T5 is now unreachable through configuration. The `_FUSED_OK`-free copy then shows `sdpa_kernel([])` **raising** "No viable backend" instead of silently running MATH. A separate run of its EFFICIENT-only case also raises. |
| `repro_guard_accounting.py` | `PER_FORWARD_REDUCTIONS [(2,)]`: one all-reduce on a non-detailed step (previously two) |
| `repro_guard_count.py` | both ranks raise the same count-mismatch error, no timeout |
| `repro_fsdp_memory.py` / `repro_fsdp_extra_wrap.py` | without `extra_wrap` the FSDP1 root is retained (upstream behaviour, as the script wraps it that way); with the pipeline's `extra_wrap`, 0 bytes in every unit on both ranks |
| `repro_review.py` / `repro_review_sections.py` | the concealed cross-attention overflow raises at `blocks.0.block.cross_attn:attn0.q`; VARLEN gives ones; 65519 is rejected; the tolerated underflow is accepted with a WARNING; CACHE_BYTES is unchanged (a synthetic `chunk()` demo that does not call the model) |
| estimator (`memory.txt`) | the default table reproduces exactly (720P 25.64 / 22.70 / 23.21 / 24.46 / 27.73 GiB) |
