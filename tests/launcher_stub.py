"""CPU/gloo stand-ins so the REAL launcher path (WanAnimate2MPIPipeline -> main -> mp.spawn ->
worker -> build_dit -> strict load + fp16 check) can run in tests without GPUs or checkpoints.

Only GPU-specific calls and the heavy T5/CLIP/VAE constructors are replaced; the worker's control
flow, status reporting, build_dit, the loader and the parent's monitoring are the real code.
Scenario via $WAN_TEST_SCENARIO: ok | die_in_inference | raise_in_inference | fail_before_result |
result_before_peer_failure | peer_death_at_completion.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class _NoDevice:
    def to(self, *a, **k):
        return self

    def clear_cache(self):
        pass


class FakeCLIP:
    def __init__(self, name, dtype, device, checkpoint_path, tokenizer_path):
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Linear(4, 4).to(dtype)
        self.model.textual = torch.nn.Linear(4, 4).to(dtype)
        self.dtype = dtype


class FakeVAE:
    def __init__(self, vae_pth, device, dtype):
        self.model = _NoDevice()
        self.scale = [_NoDevice(), _NoDevice()]


def _patch():
    from conftest import stub_heavy_video_deps
    stub_heavy_video_deps()
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.get_device_capability = lambda *a, **k: (7, 0)
    torch.cuda.memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_reserved = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda *a, **k: None
    real_init = dist.init_process_group

    def init_gloo(backend=None, **kw):
        return real_init(backend="gloo", **kw)

    dist.init_process_group = init_gloo
    from pipelines import wan_animate_2_pipeline as P
    P.CLIP = FakeCLIP
    P.VideoVAE = FakeVAE
    P.shard_transformer = lambda model, *a, **k: model
    real_build = P.build_dit
    P.build_dit = lambda cfg, device, prec, gpu: real_build(cfg, "cpu", prec, gpu)
    scenario = os.environ.get("WAN_TEST_SCENARIO", "ok")

    def fake_inference(item, gpu, *a, **k):
        # Review r2 race: both ranks finish their collectives, rank 1 then fails while rank 0
        # still produces a (late) result
        if scenario == "fail_before_result":
            dist.barrier()
            if gpu == 1:
                raise ValueError("injected failure before rank-0 result")
            time.sleep(1.6)
            return {"result": "stub-ok", "status": "success"}
        # rank 0 delivers its result first; rank 1 fails afterwards (e.g. in post-processing)
        if scenario == "result_before_peer_failure":
            dist.barrier()
            if gpu == 1:
                time.sleep(1.5)
                raise ValueError("injected failure after rank-0 result")
            return {"result": "stub-ok", "status": "success"}
        # rank 1 dies right around completion, after rank 0 delivered its result, before acking
        if scenario == "peer_death_at_completion":
            dist.barrier()
            if gpu == 1:
                time.sleep(0.5)
                os._exit(3)
            return {"result": "stub-ok", "status": "success"}
        if scenario == "die_in_inference" and gpu == 1:
            os._exit(3)
        if scenario == "raise_in_inference" and gpu == 1:
            raise ValueError("injected inference failure on gpu 1")
        dist.barrier()  # like the collectives of a real forward: ranks are coupled
        return {"result": "stub-ok", "status": "success"} if gpu == 0 else None

    P.inference_core = fake_inference
    return P


def stub_worker(gpu, *args):
    P = _patch()
    return P.worker(gpu, *args)


def tiny_launcher_cfg(ckpt_path, world=2):
    from omegaconf import OmegaConf
    from conftest import TINY
    return OmegaConf.create({
        "model": {
            "use_t5": False, "clip_model": "fake", "clip_checkpoint": "-", "clip_tokenizer": "-",
            "vae_checkpoint": "-",
            "transformer": dict(type="wanxiang.wanxiang_animate_2_arch.WanxiangAnimate2Transformer",
                                **{k: list(v) if isinstance(v, tuple) else v for k, v in TINY.items()}),
        },
        "test_cfg": {"sp_size": 1, "sharding_size": world, "sharding_strategy": "FULL_SHARD",
                     "param_dtype": "bfloat16", "reduce_dtype": "bfloat16", "buffer_dtype": "bfloat16"},
        "model_path": ckpt_path,
    })


def main():
    """Parent process: run the real pipeline constructor and one call, print what happened."""
    ckpt, port = sys.argv[1], sys.argv[2]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK="0", WORLD_SIZE="1")
    from conftest import stub_heavy_video_deps
    stub_heavy_video_deps()
    torch.cuda.device_count = lambda: 2
    from pipelines import wan_animate_2_pipeline as P
    P.worker = stub_worker  # pickled by reference: children import this module and patch themselves
    pipe = P.WanAnimate2MPIPipeline(tiny_launcher_cfg(ckpt))
    print("INIT_OK", flush=True)
    try:
        print("RESULT", pipe(refer_img_path="-", tpl_video_path="-", output_path="-"), flush=True)
    finally:
        pipe.close()
        print("CLOSED", flush=True)


if __name__ == "__main__":
    main()
