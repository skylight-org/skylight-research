#!/usr/bin/env python3
"""
Distributed mini-swe-agent + vLLM SWE-bench runner.

Replaces the OpenHands-based distributed_swebench_inference.py with a simpler stack:
  vLLM (model server) + mini-swe-agent (agent loop + Docker environment).

Quickstart (single GPU, one replica):
    conda activate swebench311
    python scripts/run_mini_swebench.py \\
        --model Qwen/Qwen3.5-27B \\
        --output benchmarks/mini/runs/my_run \\
        --num-gpus 1 \\
        --workers 4 \\
        --subset verified --split test

Multi-GPU (8 GPUs, two TP4 replicas — recommended for Qwen3.5-27B on H100s):
    python scripts/run_mini_swebench.py \\
        --model Qwen/Qwen3.5-27B \\
        --output benchmarks/mini/runs/my_run \\
        --num-gpus 8 --gpus-per-server 4 \\
        --workers 8 \\
        --subset verified --split test

Multi-GPU (8 GPUs, one TP8 replica):
    python scripts/run_mini_swebench.py \\
        --model Qwen/Qwen3.5-27B \\
        --output benchmarks/mini/runs/my_run \\
        --num-gpus 8 --gpus-per-server 8 \\
        --workers 16 \\
        --subset verified --split test

Filter to specific instances:
    python scripts/run_mini_swebench.py ... --filter '^scikit-learn__'
    python scripts/run_mini_swebench.py ... --slice 0:50

Smoke test (one instance):
    python scripts/run_mini_swebench.py ... --filter '^django__django-11099$' --workers 1

After the run:
    # Cloud eval (fast, free):
    sb-cli submit swe-bench_verified test \\
        --predictions_path benchmarks/mini/runs/my_run/preds.json \\
        --run_id my_run_id

    # Local eval:
    python -m swebench.harness.run_evaluation \\
        --dataset_name princeton-nlp/SWE-bench_Verified \\
        --predictions_path benchmarks/mini/runs/my_run/preds.json \\
        --max_workers 4 --run_id my_run_id
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_mini_swebench_config_path() -> str:
    """Packaged SWE-bench prompts + agent limits (mini-swe-agent v2+)."""
    try:
        from minisweagent.config import builtin_config_dir

        candidate = builtin_config_dir / "benchmarks" / "swebench.yaml"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not find minisweagent/config/benchmarks/swebench.yaml. "
        "Install mini-swe-agent in this environment or pass --config explicitly."
    )


def _default_litellm_registry_path() -> str:
    return str(_project_root() / "benchmarks" / "mini" / "model_registry.json")


def _mini_binary() -> str:
    """Return the mini-extra executable path inside the active conda env."""
    for candidate in ("mini-extra",):
        try:
            result = subprocess.run(
                ["which", candidate], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except OSError:
            pass
    return "mini-extra"


def _vllm_binary() -> str:
    return os.environ.get("SWEBENCH_VLLM_BIN", "vllm").strip() or "vllm"


def _server_ready_timeout() -> int:
    raw = os.environ.get("SWEBENCH_SERVER_READY_TIMEOUT", "900").strip()
    try:
        return max(60, min(int(raw), 7200))
    except ValueError:
        return 900


def wait_for_port(port: int, timeout: int, *, log_path: Optional[Path] = None) -> bool:
    """Block until localhost:port is accepting connections or timeout expires."""
    deadline = time.time() + timeout
    next_log = time.time() + 30.0
    while time.time() < deadline:
        if log_path and time.time() >= next_log:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  ... still loading ({elapsed}s/{timeout}s). See: {log_path}")
            next_log = time.time() + 30.0
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(("localhost", port)) == 0:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_servers(
    *,
    model: str,
    num_gpus: int,
    gpus_per_server: int,
    base_port: int,
    gpu_offset: int,
    output_dir: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    dtype: str,
    reasoning_parser: str,
    tool_call_parser: str,
) -> list[subprocess.Popen]:
    """Start one vLLM server per replica; each server gets gpus_per_server GPUs."""
    num_servers = num_gpus // gpus_per_server
    servers: list[subprocess.Popen] = []

    for rank in range(num_servers):
        first_gpu = gpu_offset + rank * gpus_per_server
        visible = ",".join(str(first_gpu + j) for j in range(gpus_per_server))
        port = base_port + rank
        log_file = output_dir / f"server_rank_{rank}.log"

        cmd = [
            _vllm_binary(), "serve", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--tensor-parallel-size", str(gpus_per_server),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", f"{gpu_memory_utilization:g}",
            "--dtype", dtype,
            "--trust-remote-code",
            "--language-model-only",
            "--enable-auto-tool-choice",
            "--tool-call-parser", tool_call_parser,
        ]
        if reasoning_parser and reasoning_parser != "none":
            cmd += ["--reasoning-parser", reasoning_parser]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible

        print(
            f"Starting vLLM replica {rank} on port {port} "
            f"(CUDA_VISIBLE_DEVICES={visible}) ..."
        )
        with open(log_file, "w") as fh:
            proc = subprocess.Popen(
                cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                cwd=str(_project_root()),
            )
        servers.append(proc)

        timeout = _server_ready_timeout()
        print(f"  Waiting up to {timeout}s for replica {rank} (log: {log_file}) ...")
        if wait_for_port(port, timeout, log_path=log_file):
            print(f"  ✓ Replica {rank} ready on port {port}")
        else:
            print(f"  ✗ Replica {rank} did not respond within {timeout}s — check {log_file}")

    return servers


# ---------------------------------------------------------------------------
# mini-extra swebench runner
# ---------------------------------------------------------------------------

def run_mini_workers(
    *,
    model: str,
    num_gpus: int,
    gpus_per_server: int,
    base_port: int,
    output_dir: Path,
    workers_per_replica: int,
    subset: str,
    split: str,
    filter_spec: str,
    slice_spec: str,
    config_path: Path,
    registry_path: Path,
    no_thinking: bool = False,
) -> list[subprocess.Popen]:
    """
    Launch one mini-extra swebench process per vLLM replica.

    Each process gets its own output subdirectory and points at a different port.
    Results are merged into output_dir/preds.json afterwards.
    """
    num_replicas = num_gpus // gpus_per_server
    jobs: list[subprocess.Popen] = []

    env_base = os.environ.copy()
    env_base["LITELLM_MODEL_REGISTRY_PATH"] = str(registry_path.resolve())
    env_base["MSWEA_COST_TRACKING"] = "ignore_errors"

    for rank in range(num_replicas):
        port = base_port + rank
        replica_out = output_dir / f"replica_{rank}"
        replica_out.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"mini_rank_{rank}.log"

        # hosted_vllm/<model_id> is the LiteLLM provider prefix for local vLLM
        litellm_model = f"hosted_vllm/{model}"
        api_base = f"http://127.0.0.1:{port}/v1"

        cmd = [
            _mini_binary(), "swebench",
            "--subset", subset,
            "--split", split,
            "--output", str(replica_out),
            "--workers", str(workers_per_replica),
            "-c", str(config_path.resolve()),
            f"-c", f"model.model_name={litellm_model}",
            f"-c", f"model.model_kwargs.api_base={api_base}",
        ]
        if no_thinking:
            cmd += ["-c", "model.model_kwargs.extra_body.chat_template_kwargs.enable_thinking=false"]
        if filter_spec:
            cmd += ["--filter", filter_spec]
        if slice_spec:
            cmd += ["--slice", slice_spec]

        print(f"Starting mini worker {rank} → port {port}, output: {replica_out}")
        with open(log_file, "w") as fh:
            proc = subprocess.Popen(
                cmd,
                env=env_base,
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(_project_root()),
            )
        jobs.append(proc)

    return jobs


def merge_preds(output_dir: Path, num_replicas: int) -> Path:
    """Merge replica preds.json files into a single top-level preds.json."""
    import json

    merged: dict = {}
    for rank in range(num_replicas):
        replica_preds = output_dir / f"replica_{rank}" / "preds.json"
        if replica_preds.exists():
            data = json.loads(replica_preds.read_text())
            merged.update(data)
            print(f"  ✓ Replica {rank}: {len(data)} predictions")
        else:
            print(f"  ✗ Replica {rank}: no preds.json found")

    final = output_dir / "preds.json"
    final.write_text(json.dumps(merged, indent=2))
    print(f"\nMerged {len(merged)} predictions → {final}")
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Distributed mini-swe-agent + vLLM SWE-bench runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model / server
    p.add_argument(
        "--model", required=True,
        help="HuggingFace model ID. Qwen3/3.5 are instruct by default. "
             "E.g. Qwen/Qwen3.5-27B",
    )
    p.add_argument("--num-gpus", type=int, default=1, help="Total GPUs to use (default: 1).")
    p.add_argument(
        "--gpus-per-server", type=int, default=1,
        help="GPUs per vLLM replica (tensor-parallel-size). Must divide --num-gpus. (default: 1)",
    )
    p.add_argument("--gpu-offset", type=int, default=0, help="First GPU index (default: 0).")
    p.add_argument("--base-port", type=int, default=4000, help="Port for replica 0 (default: 4000).")
    p.add_argument("--max-model-len", type=int, default=262144, help="vLLM max context length (default: 262144).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory fraction (default: 0.90).")
    p.add_argument("--dtype", default="auto", help="vLLM dtype (default: auto).")
    p.add_argument("--reasoning-parser", default="",
                   help="vLLM --reasoning-parser. Defaults to 'qwen3' for Qwen3/3.5 models, "
                        "empty (disabled) for all others. Pass 'none' to force disable.")
    p.add_argument("--tool-call-parser", default="qwen3_coder", help="vLLM --tool-call-parser (default: qwen3_coder).")

    # mini / benchmark
    p.add_argument("--output", required=True, help="Output directory for results.")
    p.add_argument("--workers", type=int, default=4,
                   help="mini-swe-agent parallel workers PER replica (default: 4). "
                        "Tune to GPU VRAM — more workers = more concurrent requests to vLLM.")
    p.add_argument("--subset", default="verified",
                   choices=["verified", "lite", "full", "multimodal", "multilingual"],
                   help="SWE-bench subset (default: verified).")
    p.add_argument("--split", default="test", help="Dataset split (default: test).")
    p.add_argument("--filter", default="", dest="filter_spec",
                   help="Regex filter on instance IDs. E.g. '^django__django'")
    p.add_argument("--slice", default="", dest="slice_spec",
                   help="Slice specification, e.g. '0:50' for first 50 instances.")

    # Paths
    p.add_argument(
        "--config",
        default="",
        help=(
            "Path to mini config YAML (default: minisweagent packaged "
            "benchmarks/swebench.yaml)."
        ),
    )
    p.add_argument(
        "--registry",
        default="",
        help=(
            "Path to LiteLLM model registry JSON (default: "
            "benchmarks/mini/model_registry.json, created empty if missing)."
        ),
    )

    # Control
    p.add_argument("--skip-server", action="store_true",
                   help="Skip starting vLLM (assume server already running on --base-port).")
    p.add_argument("--skip-validation", action="store_true", help="Skip pre-flight checks.")
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable thinking mode (set enable_thinking=false). Use for Qwen2.5 and "
                        "other models that do not support the thinking chat template kwarg.")

    args = p.parse_args()

    if not str(args.config).strip():
        try:
            args.config = _default_mini_swebench_config_path()
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    if not str(args.registry).strip():
        args.registry = _default_litellm_registry_path()

    if args.num_gpus % args.gpus_per_server != 0:
        print(
            f"Error: --num-gpus ({args.num_gpus}) must be divisible by "
            f"--gpus-per-server ({args.gpus_per_server})"
        )
        sys.exit(1)

    # Auto-detect reasoning parser: Qwen3/3.5 use qwen3 parser; Qwen2.5 and others do not.
    if not args.reasoning_parser:
        model_lower_tmp = args.model.lower()
        args.reasoning_parser = "qwen3" if "qwen3" in model_lower_tmp else ""

    # Auto-enable --no-thinking for non-Qwen3 models (Qwen2.5 doesn't support the kwarg).
    if not args.no_thinking and "qwen3" not in args.model.lower():
        args.no_thinking = True

    # Enforce instruct/chat model.
    # Qwen3 and Qwen3.5 are instruction-tuned by default (no -Instruct suffix needed).
    # Qwen2.5 and earlier require the explicit -Instruct suffix.
    model_lower = args.model.lower()
    _instruct_keywords = ("instruct", "coder", "chat")
    _always_instruct_prefixes = ("qwen/qwen3",)  # Qwen3, Qwen3.5, Qwen3-Coder all instruction-tuned
    is_always_instruct = any(p in model_lower for p in _always_instruct_prefixes)
    has_instruct_kw = any(kw in model_lower for kw in _instruct_keywords)
    if not (is_always_instruct or has_instruct_kw):
        print(
            f"Error: model '{args.model}' does not appear to be an instruct/chat variant.\n"
            "This project uses instruct models only.\n"
            "Examples of accepted models:\n"
            "  Qwen/Qwen3.5-27B          (Qwen3.5 — instruct by default)\n"
            "  Qwen/Qwen3-32B             (Qwen3 — instruct by default)\n"
            "  Qwen/Qwen3-Coder-30B-A3B-Instruct\n"
            "  Qwen/Qwen2.5-72B-Instruct  (Qwen2.5 requires explicit -Instruct suffix)"
        )
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        registry_path.write_text("{}\n", encoding="utf-8")
    num_replicas = args.num_gpus // args.gpus_per_server

    if not args.skip_validation:
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}")
            sys.exit(1)
        if not registry_path.exists():
            print(f"Error: model registry not found: {registry_path}")
            sys.exit(1)
        try:
            subprocess.run([_mini_binary(), "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: mini-extra not found. Run: conda activate swebench311 && pip install mini-swe-agent")
            sys.exit(1)

    print("=" * 70)
    print("mini-swe-agent + vLLM distributed SWE-bench runner")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  GPUs:         {args.num_gpus} total, {args.gpus_per_server}/server → {num_replicas} replica(s)")
    print(f"  Ports:        {args.base_port} – {args.base_port + num_replicas - 1}")
    print(f"  Workers/rep:  {args.workers}")
    print(f"  Dataset:      {args.subset}/{args.split}"
          + (f"  filter={args.filter_spec}" if args.filter_spec else "")
          + (f"  slice={args.slice_spec}" if args.slice_spec else ""))
    print(f"  Output:       {output_dir}")
    print("=" * 70)

    servers: list[subprocess.Popen] = []
    mini_jobs: list[subprocess.Popen] = []

    try:
        # Step 1: Start vLLM servers
        if args.skip_server:
            print("\n[skip] vLLM server startup (--skip-server)")
        else:
            print("\n=== Step 1: Starting vLLM servers ===")
            servers = start_vllm_servers(
                model=args.model,
                num_gpus=args.num_gpus,
                gpus_per_server=args.gpus_per_server,
                base_port=args.base_port,
                gpu_offset=args.gpu_offset,
                output_dir=output_dir,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                reasoning_parser=args.reasoning_parser,
                tool_call_parser=args.tool_call_parser,
            )

        # Step 2: Start mini workers
        print("\n=== Step 2: Starting mini-swe-agent workers ===")
        mini_jobs = run_mini_workers(
            model=args.model,
            num_gpus=args.num_gpus,
            gpus_per_server=args.gpus_per_server,
            base_port=args.base_port,
            output_dir=output_dir,
            workers_per_replica=args.workers,
            subset=args.subset,
            split=args.split,
            filter_spec=args.filter_spec,
            slice_spec=args.slice_spec,
            config_path=config_path,
            registry_path=registry_path,
            no_thinking=args.no_thinking,
        )

        # Step 3: Wait for mini workers to finish
        print("\n=== Step 3: Waiting for mini workers ===")
        while True:
            all_done = True
            for rank, job in enumerate(mini_jobs):
                if job.poll() is None:
                    all_done = False
                    print(f"  Replica {rank} mini worker: running (PID {job.pid})")
                else:
                    status = "✓" if job.returncode == 0 else f"✗ exit={job.returncode}"
                    print(f"  Replica {rank} mini worker: {status}")
            if all_done:
                break
            time.sleep(30)

        # Step 4: Merge preds
        print("\n=== Step 4: Merging predictions ===")
        final_preds = merge_preds(output_dir, num_replicas)

        print("\n=== Done! ===")
        print(f"  Predictions: {final_preds}")
        print(f"  Trajectories: {output_dir}/replica_*/")
        print()
        print("Next steps — evaluate with sb-cli (fast, free cloud eval):")
        print(f"  sb-cli submit swe-bench_verified {args.split} \\")
        print(f"    --predictions_path {final_preds} \\")
        print(f"    --run_id <your_run_id>")
        print()
        print("Or local evaluation:")
        print(f"  python -m swebench.harness.run_evaluation \\")
        print(f"    --dataset_name princeton-nlp/SWE-bench_Verified \\")
        print(f"    --predictions_path {final_preds} \\")
        print(f"    --max_workers 4 --run_id <your_run_id>")

    except KeyboardInterrupt:
        print("\nInterrupted — shutting down ...")

    finally:
        print("\n=== Cleanup: terminating vLLM servers ===")
        for rank, srv in enumerate(servers):
            if srv.poll() is None:
                srv.terminate()
                try:
                    srv.wait(timeout=10)
                    print(f"  ✓ Replica {rank} server terminated")
                except subprocess.TimeoutExpired:
                    srv.kill()
                    print(f"  ✓ Replica {rank} server killed")
            else:
                print(f"  Replica {rank} server already stopped")


if __name__ == "__main__":
    main()
