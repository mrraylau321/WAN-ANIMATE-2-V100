"""2-process gloo workers for the loader / guard tests (spawned by mp.spawn; CPU only)."""
import json
import os
import traceback
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist


@contextmanager
def _cpu_cuda_fp32_regions():
    """Emulate nested CUDA FP32 autocast-off blocks during CPU FP16 smoke tests."""
    real = torch.amp.autocast

    def shim(device_type, *args, **kwargs):
        if device_type == "cuda" and kwargs.get("dtype") == torch.float32:
            return real("cpu", enabled=False)
        return real(device_type, *args, **kwargs)

    torch.amp.autocast = shim
    try:
        yield
    finally:
        torch.amp.autocast = real


def _init(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def tiny_cfg(ckpt_path, world, fp32_modules=(), dit_dtype="fp16", shard_dtype="auto"):
    from omegaconf import OmegaConf
    from conftest import TINY
    return OmegaConf.create({
        "model": {"transformer": dict(type="wanxiang.wanxiang_animate_2_arch.WanxiangAnimate2Transformer",
                                      **{k: list(v) if isinstance(v, tuple) else v for k, v in TINY.items()})},
        "test_cfg": {"sharding_size": world, "world_size": world, "sp_size": 1,
                     "fp32_modules": list(fp32_modules), "dit_dtype": dit_dtype,
                     "dit_shard_dtype": shard_dtype},
        "model_path": ckpt_path,
    })


def load_worker(rank, world, port, scenario, ckpt_path, out_dir):
    """Run pipelines.wan_animate_2_pipeline.build_dit (the pipeline's own loader) on CPU/gloo."""
    from conftest import stub_heavy_video_deps, make_forward_inputs, run_model
    stub_heavy_video_deps()
    res = {"rank": rank, "scenario": scenario}
    try:
        _init(rank, world, port)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        from pipelines import wan_animate_2_pipeline as P
        from wanxiang.models import attention as A
        from wanxiang.utils.precision import resolve_precision
        fp32_modules = ["blocks.1"] if scenario in ("overflow_allowlisted", "shard_bf16_island") else []
        if scenario == "shard_bf16_fp16_forward":
            fp32_modules = ["time_embedding", "time_projection", "head"]
        dit_dtype = "fp32" if scenario in ("forward_fp32", "shard_bf16_forward") else "fp16"
        shard_dtype = "bf16" if scenario.startswith("shard_bf16") else "auto"
        cfg = tiny_cfg(ckpt_path, world, fp32_modules, dit_dtype, shard_dtype)
        prec = resolve_precision(cfg.test_cfg, torch.device("cpu"))
        A.set_attention_backend("sdpa")
        A.set_compute_dtype(prec.dit)
        model, fp32 = P.build_dit(cfg, "cpu", prec, gpu=rank)
        res["prec"] = prec.describe()
        p = dict(model.named_parameters())["blocks.1.block.ffn.0.weight"]
        res["param_type"] = type(p).__name__
        res["local_rows"] = int(p.to_local().shape[0])
        res["storage_dtype"] = str(p.dtype)
        res["other_storage_dtype"] = str(dict(model.named_parameters())["blocks.0.block.ffn.0.weight"].dtype)
        res["local_allocation"] = P.local_weight_allocation(model, "cpu")
        res["fp32_units"] = [n for n, m in model.named_modules() if getattr(m, "_wan_fp32", False)]
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            torch.save({k: v.clone() for k, v in full.items()}, os.path.join(out_dir, f"{scenario}_sd.pt"))
        if scenario in ("forward_fp32", "shard_bf16_forward", "shard_bf16_fp16_forward"):
            use_fp16 = scenario == "shard_bf16_fp16_forward"
            with torch.no_grad(), (_cpu_cuda_fp32_regions() if use_fp16 else nullcontext()), \
                    torch.autocast(device_type="cpu", dtype=torch.float16, enabled=use_fp16):
                out = run_model(model, make_forward_inputs())[0]
            torch.save(out, os.path.join(out_dir, f"{scenario}_out_rank{rank}.pt"))
        res["ok"] = True
    except Exception as e:  # report instead of hanging the parent
        res["ok"] = False
        res["error_type"] = type(e).__name__
        res["error"] = str(e)[:4000]
        res["tb"] = traceback.format_exc()[-3000:]
    finally:
        with open(os.path.join(out_dir, f"{scenario}_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()


def guard_worker(rank, world, port, out_dir):
    """Rank 1 alone sees a NaN; both ranks must agree on the finding (no deadlock)."""
    from conftest import TINY, make_forward_inputs, run_model
    res = {"rank": rank}
    try:
        _init(rank, world, port)
        from wanxiang.models import attention as A
        from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer
        from wanxiang.utils.fp16_guard import OverflowGuard, FP16OverflowError
        A.set_attention_backend("sdpa")
        A.set_compute_dtype(torch.float32)
        torch.manual_seed(0)
        model = WanAnimate2Transformer(**TINY).eval()
        for prm in model.parameters():
            prm.data.normal_(0, 0.05)
        if rank == 1:
            model.blocks[2].block.ffn[2].bias.data[3] = float("nan")
        guard = OverflowGuard(model, mode="first_step", action="abort")
        try:
            with torch.no_grad():
                run_model(model, make_forward_inputs(), call=lambda f, s, b: guard.run(f, s, b))
            res["raised"] = False
        except FP16OverflowError as e:
            res["raised"] = True
            res["msg"] = str(e)
    except Exception as e:
        res["error"] = traceback.format_exc()[-3000:]
    finally:
        with open(os.path.join(out_dir, f"guard_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()


def guard_count_worker(rank, world, port, out_dir):
    """Review F6 repro: rank r holds r+1 records; every rank must raise the same error."""
    from datetime import timedelta
    res = {"rank": rank}
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=20))
        from wanxiang.utils.fp16_guard import OverflowGuard
        g = OverflowGuard(torch.nn.Linear(1, 1), mode="all")
        g._records = [("head[0]", "float32", torch.tensor(1.))] * (rank + 1)
        try:
            g._check(torch.ones(1), 0, "cond")
            res["error_type"] = None
        except Exception as e:
            res["error_type"] = type(e).__name__
            res["error"] = str(e)
    finally:
        with open(os.path.join(out_dir, f"count_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()


def assert_finite_worker(rank, world, port, out_dir):
    res = {"rank": rank, "raised": False}
    try:
        _init(rank, world, port)
        from wanxiang.utils.fp16_guard import OverflowGuard, FP16OverflowError
        g = OverflowGuard(torch.nn.Linear(1, 1), mode="first_step")
        x = torch.ones(4)
        if rank == 1:
            x[2] = float("nan")
        try:
            g.assert_finite("conditioning", {"VAE latents (reference image)": torch.ones(3),
                                             "CLIP visual (reference image)": [x, torch.zeros(())]},
                            hints={"CLIP visual (reference image)": "set test_cfg.clip_dtype=fp32"})
        except FP16OverflowError as e:
            res["raised"], res["msg"] = True, str(e)
    except Exception:
        res["error"] = traceback.format_exc()[-3000:]
    finally:
        with open(os.path.join(out_dir, f"finite_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()


def cp_worker(rank, world, port, out_dir):
    """Context-parallel forward (sp=world) of the tiny model on gloo: output + K/V cache storage."""
    from conftest import TINY, make_forward_inputs
    res = {"rank": rank}
    try:
        _init(rank, world, port)

        def all_to_all_via_gather(outputs, inputs, group=None, **kw):
            # gloo has no all_to_all; same semantics via all_gather (test-only emulation)
            stacked = torch.stack([t.contiguous() for t in inputs])
            gathered = [torch.empty_like(stacked) for _ in range(dist.get_world_size(group))]
            dist.all_gather(gathered, stacked, group=group)
            me = dist.get_rank(group)
            for j, out in enumerate(outputs):
                out.copy_(gathered[j][me])

        dist.all_to_all = all_to_all_via_gather
        from wanxiang import ops
        from wanxiang.models import attention as A
        from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer
        ops.init_model_parallel_groups(world)
        A.set_attention_backend("sdpa")
        A.set_compute_dtype(torch.float32)
        torch.manual_seed(0)
        model = WanAnimate2Transformer(**dict(TINY, use_context_parallel=True)).eval()
        for prm in model.parameters():
            prm.data.normal_(0, 0.05)
        inp = make_forward_inputs()
        k_cache, v_cache = {}, {}
        with torch.no_grad():
            model(inp["cond_lat"], grid_sizes=inp["grid_sizes"], k_cache=k_cache, v_cache=v_cache,
                  t=inp["t"], method="forward_ref", context_ref=inp["ctx_ref"],
                  seq_len_ref=inp["seq_len_ref"], clip_fea_ref=inp["clip_fea_ref"], y_ref=inp["cond_y"])
            res["cache"] = [[k.untyped_storage().nbytes(), k.numel() * k.element_size(),
                             v.untyped_storage().nbytes(), v.numel() * v.element_size(),
                             k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()]
                            for k, v in ((k_cache[i], v_cache[i]) for i in sorted(k_cache))]
            res["cache_shape"] = list(k_cache[0].shape)
            out = model(inp["noise"], k_cache=k_cache, v_cache=v_cache, t=inp["t"],
                        grid_sizes_ref=inp["grid_sizes_ref"], method="forward_gen", context=inp["ctx"],
                        seq_len=inp["seq_len"], clip_fea=inp["clip_fea"], y=inp["y"],
                        origin_len=inp["clip_len"], origin_area=inp["area"])[0]
        torch.save(out, os.path.join(out_dir, f"cp_out_rank{rank}.pt"))
        res["ok"] = True
    except Exception:
        res["ok"] = False
        res["tb"] = traceback.format_exc()[-3000:]
    finally:
        with open(os.path.join(out_dir, f"cp_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()


def t5_worker(rank, world, port, out_dir):
    """T5 sharded exactly as the pipeline does on the V100 path (FSDP1, bf16 storage, fp32
    compute, token embedding + final norm as extra units) vs an fp32 reference."""
    import copy
    res = {"rank": rank}
    try:
        _init(rank, world, port)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from wanxiang.distributed.fsdp import shard_model
        from wanxiang.models.t5 import _t5
        kw = dict(vocab_size=100, dim=32, dim_attn=32, dim_ffn=64, num_heads=4, encoder_layers=2,
                  decoder_layers=2, num_buckets=8, shared_pos=False, dropout=0.0)
        torch.manual_seed(0)
        t5 = _t5("umt5-tiny", encoder_only=True, dtype=torch.bfloat16, **kw).eval().requires_grad_(False)
        sd = copy.deepcopy(t5.state_dict())
        fresh = _t5("umt5-tiny", encoder_only=True, dtype=torch.bfloat16, **kw)
        fresh.load_state_dict(sd)  # strict default, as T5Encoder does
        ref = copy.deepcopy(t5).float()
        res["param_dtypes"] = sorted({str(p.dtype) for p in t5.parameters()})
        sharded = shard_model(t5, device_id=torch.device("cpu"), param_dtype=torch.float32,
                              reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
                              sync_module_states=False, sharding_strategy=ShardingStrategy.FULL_SHARD,
                              extra_wrap=(t5.token_embedding, t5.norm))
        ids = torch.randint(0, 100, (1, 12))
        mask = torch.ones(1, 12, dtype=torch.long)
        mask[0, 9:] = 0
        with torch.no_grad():
            got, want = sharded(ids, mask), ref(ids, mask)
        res["max_diff"] = float((got - want).abs().max())
        res["out_dtype"] = str(got.dtype)
        res["retained"] = {n or "<root>": m._handle.flat_param._full_param_padded.untyped_storage().nbytes()
                           for n, m in sharded.named_modules() if isinstance(m, FSDP) and m._handle is not None}
        res["local_dtype"] = sorted({str(m._handle.flat_param._local_shard.dtype)
                                     for m in sharded.modules() if isinstance(m, FSDP) and m._handle is not None})
        res["keys_equal"] = {k.replace("_fsdp_wrapped_module.", "") for k in sharded.state_dict()} == set(sd)
        res["ok"] = True
    except Exception:
        res["ok"] = False
        res["tb"] = traceback.format_exc()[-3000:]
    finally:
        with open(os.path.join(out_dir, f"t5_rank{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            dist.destroy_process_group()
