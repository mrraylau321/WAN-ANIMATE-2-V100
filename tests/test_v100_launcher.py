"""Review F2: failures anywhere in the workers end the launcher with a clear error and a
non-zero exit instead of waiting forever. Runs the real launcher path on CPU/gloo (see
launcher_stub.py) in a subprocess with a hard timeout."""
import os
import subprocess
import sys
import time

import pytest

from conftest import REPO
from test_v100_state_dict import _free_port, _save_bf16_ckpt

TIMEOUT = 180


def _launch(tmp_path, ckpt, scenario="ok"):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO, os.path.join(REPO, "tests")]),
               WAN_TEST_SCENARIO=scenario, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(REPO, "tests", "launcher_stub.py"), ckpt,
                            str(_free_port())], capture_output=True, text=True, timeout=TIMEOUT, env=env,
                           cwd=str(tmp_path))
    except subprocess.TimeoutExpired as e:  # the bug this test guards against
        pytest.fail(f"launcher hung for {TIMEOUT}s:\n{(e.stdout or b'')[-3000:]}")
    return p, time.time() - t0


def test_load_rejection_exits_nonzero_with_original_error(tmp_path):
    ckpt = str(tmp_path / "ovf.safetensors")

    def mutate(sd):
        sd["blocks.1.block.ffn.0.weight"][3, 2] = 1e6

    _save_bf16_ckpt(ckpt, mutate)
    p, dt = _launch(tmp_path, ckpt)
    out = p.stdout + p.stderr
    assert p.returncode != 0, out[-3000:]
    assert "INIT_OK" not in out
    assert "Wan-Animate-2 initialisation failed" in out
    assert "FP16OverflowError" in out and "blocks.1.block.ffn.0: 1 values exceed the fp16 range" in out


def test_clean_run_returns_result_and_exits(tmp_path):
    ckpt = str(tmp_path / "ok.safetensors")
    _save_bf16_ckpt(ckpt)
    p, dt = _launch(tmp_path, ckpt)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "INIT_OK" in p.stdout and "RESULT stub-ok" in p.stdout and "CLOSED" in p.stdout


@pytest.mark.parametrize("scenario,expect", [
    ("die_in_inference", "worker(s) exited without a report"),
    ("raise_in_inference", "worker 1: ValueError: injected inference failure on gpu 1"),
    # Review r2 N1 (r2_artifacts/launcher_race.py): failure reported before rank 0's result
    ("fail_before_result", "worker 1: ValueError: injected failure before rank-0 result"),
    ("result_before_peer_failure", "worker 1: ValueError: injected failure after rank-0 result"),
    ("peer_death_at_completion", "(1, 3)"),
])
def test_worker_failure_during_inference_raises(tmp_path, scenario, expect):
    ckpt = str(tmp_path / "ok.safetensors")
    _save_bf16_ckpt(ckpt)
    p, dt = _launch(tmp_path, ckpt, scenario)
    out = p.stdout + p.stderr
    assert p.returncode != 0, out[-3000:]
    assert "INIT_OK" in p.stdout and "RESULT" not in p.stdout and "CLOSED" in p.stdout
    assert "Wan-Animate-2 inference failed" in out and expect in out, out[-3000:]
