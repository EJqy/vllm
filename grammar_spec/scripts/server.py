"""Launch / stop a vLLM server for one baseline configuration and record what
it actually ran with (model runner version, async scheduling, spec config).

Usage (CLI):
  .venv/bin/python grammar_spec/scripts/server.py start --method eagle3 --k 5 \
      --ws compact [--sync] [--runner v1] [--run-id NAME]
  .venv/bin/python grammar_spec/scripts/server.py stop

The server is bound to 127.0.0.1 with VLLM_SERVER_DEV_MODE=1 so that
/server_info (resolved config) and /reset_prefix_cache are available.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "grammar_spec" / "results" / "logs"
PID_FILE = REPO / "grammar_spec" / ".server.pid"  # outside results/ (append-only)

MODEL = "Qwen/Qwen3-32B"
EAGLE3 = "RedHatAI/Qwen3-32B-speculator.eagle3"
COMMON_ARGS = ["--quantization", "fp8_per_tensor", "--max-model-len", "8192"]


def spec_config(method: str, k: int) -> dict | None:
    if method == "none":
        return None
    if method == "eagle3":
        return {"method": "eagle3", "model": EAGLE3, "num_speculative_tokens": k}
    if method == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": k,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
        }
    if method == "suffix":
        return {"method": "suffix", "num_speculative_tokens": k}
    raise ValueError(method)


def run_id_for(method, k, ws, sync, runner, grammar_draft=None):
    rid = f"{method}" + (f"_k{k}" if method != "none" else "") + f"_{ws}"
    if sync:
        rid += "_sync"
    if runner != "auto":
        rid += f"_{runner}"
    if grammar_draft:
        rid += {"hybrid": "_gd", "grammar_only": "_gdonly"}[grammar_draft]
    return rid


def build_cmd(method, k, ws, sync, port, extra=()):
    cmd = [str(REPO / ".venv" / "bin" / "vllm"), "serve", MODEL, *COMMON_ARGS]
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    sc = spec_config(method, k)
    if sc is not None:
        cmd += ["--speculative-config", json.dumps(sc)]
    so = {"backend": "xgrammar", "disable_any_whitespace": ws == "compact"}
    cmd += ["--structured-outputs-config", json.dumps(so)]
    if sync:
        cmd += ["--no-async-scheduling"]
    cmd += list(extra)
    return cmd


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def wait_healthy(port, proc, timeout_s=1200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            _get(f"http://127.0.0.1:{port}/health", timeout=2)
            return time.time() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(3)
    raise TimeoutError("server did not become healthy")


def inspect_server(port, log_path):
    """Return what the running server actually uses."""
    log = Path(log_path).read_text(errors="replace")
    info = json.loads(_get(f"http://127.0.0.1:{port}/server_info?config_format=json"))
    cfg = info["vllm_config"]
    sched = cfg.get("scheduler_config", {})
    spec = cfg.get("speculative_config") or {}
    so = cfg.get("structured_outputs_config", {})
    return {
        "model_runner": "V2" if "Using V2 Model Runner" in log else "V1",
        "v2_fallback_warning": next(
            (ln for ln in log.splitlines() if "Model Runner V2 does not yet" in ln),
            None,
        ),
        "async_scheduling": sched.get("async_scheduling"),
        "async_disabled_warning": next(
            (ln for ln in log.splitlines() if "Async scheduling is disabled" in ln),
            None,
        ),
        "max_num_seqs": sched.get("max_num_seqs"),
        "max_num_batched_tokens": sched.get("max_num_batched_tokens"),
        "spec_method": spec.get("method"),
        "num_speculative_tokens": spec.get("num_speculative_tokens"),
        "draft_sample_method": spec.get("draft_sample_method"),
        "disable_any_whitespace": so.get("disable_any_whitespace"),
        "so_backend": so.get("backend"),
        "kv_cache_line": next(
            (ln for ln in log.splitlines() if "GPU KV cache size" in ln), None
        ),
    }


class Server:
    """Context manager: start a server, inspect it, stop it on exit."""

    def __init__(self, method, k=5, ws="compact", sync=False, runner="auto",
                 port=8000, run_id=None, extra=(), grammar_draft=None):
        """`grammar_draft`: None, "hybrid" or "grammar_only" enables the
        vllm-grammar-draft plugin (which requires synchronous scheduling)."""
        if grammar_draft and not sync:
            raise ValueError("the grammar-draft plugin needs sync=True")
        self.method, self.k, self.ws, self.sync = method, k, ws, sync
        self.runner, self.port, self.extra = runner, port, extra
        self.grammar_draft = grammar_draft
        self.run_id = run_id or run_id_for(method, k, ws, sync, runner,
                                           grammar_draft)
        self.log_path = LOG_DIR / f"serve_{self.run_id}.log"
        self.gd_stats_path = (
            LOG_DIR / f"serve_{self.run_id}.gd_stats.json" if grammar_draft else None
        )
        self.proc = None
        self.meta = None

    def start(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, VLLM_SERVER_DEV_MODE="1")
        if self.runner != "auto":
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if self.runner == "v2" else "0"
        env.pop("GRAMMAR_DRAFT", None)
        if self.grammar_draft:
            env.update(GRAMMAR_DRAFT="1", GRAMMAR_DRAFT_MODE=self.grammar_draft,
                       GRAMMAR_DRAFT_STATS=str(self.gd_stats_path))
        cmd = build_cmd(self.method, self.k, self.ws, self.sync, self.port,
                        self.extra)
        with open(self.log_path, "w") as log:
            log.write("# " + " ".join(cmd) + "\n")
            log.flush()
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True, cwd=REPO,
            )
        PID_FILE.write_text(str(self.proc.pid))
        startup_s = wait_healthy(self.port, self.proc)
        log = self.log_path.read_text(errors="replace")
        self.meta = {
            "run_id": self.run_id,
            "cmd": cmd,
            "env": {k: env[k] for k in env
                    if k.startswith(("VLLM_", "GRAMMAR_DRAFT"))},
            "startup_s": round(startup_s, 1),
            "grammar_draft": self.grammar_draft,
            "grammar_draft_active": "Grammar draft enabled" in log,
            "gd_stats_path": str(self.gd_stats_path) if self.gd_stats_path else None,
            **inspect_server(self.port, self.log_path),
        }
        if bool(self.grammar_draft) != self.meta["grammar_draft_active"]:
            raise RuntimeError(f"grammar draft plugin state mismatch: {self.meta}")
        meta_path = self.log_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.meta, indent=2))
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        PID_FILE.unlink(missing_ok=True)

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *exc):
        self.stop()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--method", choices=["none", "eagle3", "ngram", "suffix"],
                   required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--ws", choices=["default", "compact"], default="compact")
    s.add_argument("--sync", action="store_true")
    s.add_argument("--runner", choices=["auto", "v1", "v2"], default="auto")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--run-id", default=None)
    s.add_argument("--grammar-draft", choices=["hybrid", "grammar_only"],
                   default=None)
    sub.add_parser("stop")
    args = p.parse_args()

    if args.cmd == "stop":
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            PID_FILE.unlink(missing_ok=True)
            print(f"sent SIGTERM to process group {pid}")
        return

    srv = Server(args.method, args.k, args.ws, args.sync, args.runner,
                 args.port, args.run_id, grammar_draft=args.grammar_draft)
    try:
        srv.start()
    except BaseException:
        srv.stop()
        raise
    print(json.dumps(srv.meta, indent=2))
    sys.stdout.flush()
    # Detach: leave the server running; `server.py stop` kills it.
    os._exit(0)


if __name__ == "__main__":
    main()
