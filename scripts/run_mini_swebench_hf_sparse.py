#!/usr/bin/env python3
"""
Simple mini-swe-agent runner using the local HuggingFace sparse-attention server.

This script does three things:
1. Starts ``scripts/start_server.py`` with the repo's sparse OracleTopK(20%) config.
2. Runs ``mini-extra swebench`` against that local OpenAI-compatible endpoint.
3. Writes sparse micro-metrics into ``<output>/micro_metrics/micro_metrics.jsonl``.

Default behavior is intentionally sample-friendly:
- subset: ``lite``
- split: ``test``
- slice: ``0:5`` if neither ``--filter`` nor ``--slice`` is provided
- workers: ``1``

Example:
    python scripts/run_mini_swebench_hf_sparse.py \
        --output results/mini_swe/sample

Specific sample set:
    python scripts/run_mini_swebench_hf_sparse.py \
        --output results/mini_swe/pytest_sample \
        --filter '^pytest-dev__pytest-' \
        --workers 2

If you want to pin GPUs explicitly:
    python scripts/run_mini_swebench_hf_sparse.py \
        --output results/mini_swe/sample \
        --visible-gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_server_python() -> str:
    for key in ("SPARSE_ATTENTION_SERVER_PYTHON", "SWEBENCH_SERVER_PYTHON"):
        candidate = os.environ.get(key, "").strip()
        if candidate:
            return candidate
    return sys.executable


def _wait_for_port(port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


def _run_check(cmd: list[str], error_message: str) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SystemExit(f"{error_message}\nOS error: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            raise SystemExit(f"{error_message}\n{detail}")
        raise SystemExit(error_message)


def _default_is_hybrid(model: str) -> bool:
    model_lower = model.lower()
    return "qwen3" in model_lower


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent samples with the local HF sparse backend."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-27B",
        help="HuggingFace model ID for the sparse server.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for preds, logs, and micro_metrics.",
    )
    parser.add_argument(
        "--subset",
        default="lite",
        choices=["verified", "lite", "full", "multimodal", "multilingual"],
        help="SWE-bench subset. Default is lite for sample runs.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="SWE-bench split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="mini-swe-agent parallel workers.",
    )
    parser.add_argument(
        "--filter",
        default="",
        dest="filter_spec",
        help="Regex filter on instance IDs.",
    )
    parser.add_argument(
        "--slice",
        default="",
        dest="slice_spec",
        help="Slice like 0:5. If omitted and --filter is empty, defaults to 0:5.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Local port for the sparse server.",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=1200,
        help="Seconds to wait for model load.",
    )
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=float,
        default=70.0,
        help="Per-visible-GPU cap passed to start_server.py when using device_map=auto.",
    )
    parser.add_argument(
        "--visible-gpus",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value for the server process, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--server-python",
        default="",
        help="Optional Python executable for scripts/start_server.py.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=true to the client config. Default keeps thinking off.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Do not pass --is-hybrid to start_server.py.",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override mini-swe-agent agent.step_limit (packaged swebench.yaml default is 250). "
            "Lower values cut cost/time but raise risk the agent hits the limit before submitting."
        ),
    )
    args = parser.parse_args()

    if args.filter_spec and args.slice_spec:
        raise SystemExit("Use either --filter or --slice, not both.")

    if not args.filter_spec and not args.slice_spec:
        args.slice_spec = "0:5"

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    micro_metrics_dir = output_dir / "micro_metrics"
    micro_metrics_dir.mkdir(parents=True, exist_ok=True)

    server_log = output_dir / "server.log"
    mini_log = output_dir / "mini.log"

    _run_check(
        ["docker", "info"],
        "Docker is required and does not appear to be running.",
    )
    _run_check(
        ["mini-extra", "--help"],
        "mini-extra was not found. Install mini-swe-agent in this environment.",
    )

    server_python = args.server_python.strip() or _resolve_server_python()
    project_root = _project_root()
    api_base = f"http://127.0.0.1:{args.port}/v1"
    client_model_name = f"openai/{args.model}"

    config_data = {
        "model": {
            "model_name": client_model_name,
            "model_kwargs": {
                "api_base": api_base,
                "api_key": "not-needed",
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 16384,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": bool(args.enable_thinking),
                    }
                },
            },
        }
    }

    server_env = os.environ.copy()
    server_env["SAH_METRICS_LOG_DIR"] = str(micro_metrics_dir)
    if args.visible_gpus.strip():
        server_env["CUDA_VISIBLE_DEVICES"] = args.visible_gpus.strip()

    server_cmd = [
        server_python,
        str(project_root / "scripts" / "start_server.py"),
        "--device-map",
        "auto",
        "--max-memory-per-gpu-gib",
        f"{args.max_memory_per_gpu_gib:g}",
    ]
    if _default_is_hybrid(args.model) and not args.no_hybrid:
        server_cmd.append("--is-hybrid")
    server_cmd.extend([args.model, str(args.port)])

    mini_cmd = [
        "mini-extra",
        "swebench",
        "--subset",
        args.subset,
        "--split",
        args.split,
        "--output",
        str(output_dir),
        "--workers",
        str(args.workers),
    ]

    server_proc: subprocess.Popen[str] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="mini_swe_hf_sparse_") as tmp_dir:
            # mini-swe-agent v2: any -c replaces the default; we must merge the
            # packaged SWE-bench agent templates before the local model overlay.
            from minisweagent.config import builtin_config_dir

            swebench_agent_config = builtin_config_dir / "benchmarks" / "swebench.yaml"
            if not swebench_agent_config.is_file():
                raise SystemExit(
                    "Could not find minisweagent SWE-bench config at "
                    f"{swebench_agent_config}. Is mini-swe-agent installed?"
                )

            tmp_path = Path(tmp_dir)
            cfg_path = tmp_path / "mini_config.yaml"
            registry_path = tmp_path / "model_registry.json"
            cfg_path.write_text(
                yaml.safe_dump(
                    config_data,
                    sort_keys=False,
                    default_flow_style=False,
                ),
                encoding="utf-8",
            )
            registry_path.write_text("{}", encoding="utf-8")

            mini_cmd.extend(["-c", str(swebench_agent_config)])
            mini_cmd.extend(["-c", str(cfg_path)])
            mini_cmd.extend(["-c", f"model.model_name={client_model_name}"])
            mini_cmd.extend(["-c", f"model.model_kwargs.api_base={api_base}"])
            if args.step_limit is not None:
                mini_cmd.extend(["-c", f"agent.step_limit={args.step_limit}"])
            if args.filter_spec:
                mini_cmd.extend(["--filter", args.filter_spec])
            if args.slice_spec:
                mini_cmd.extend(["--slice", args.slice_spec])

            mini_env = os.environ.copy()
            mini_env["LITELLM_MODEL_REGISTRY_PATH"] = str(registry_path)
            mini_env["MSWEA_COST_TRACKING"] = "ignore_errors"

            print("=" * 72)
            print("HF sparse mini-swe-agent sample runner")
            print("=" * 72)
            print(f"Model          : {args.model}")
            print(f"Server Python  : {server_python}")
            print(f"API Base       : {api_base}")
            print(f"Subset/Split   : {args.subset}/{args.split}")
            print(f"Workers        : {args.workers}")
            if args.filter_spec:
                print(f"Filter         : {args.filter_spec}")
            if args.slice_spec:
                print(f"Slice          : {args.slice_spec}")
            if args.visible_gpus.strip():
                print(f"Visible GPUs   : {args.visible_gpus.strip()}")
            if args.step_limit is not None:
                print(f"Step limit     : {args.step_limit}")
            print(f"Output         : {output_dir}")
            print("=" * 72)

            print("\nStarting sparse HF server...")
            with server_log.open("w", encoding="utf-8") as server_fh:
                server_proc = subprocess.Popen(
                    server_cmd,
                    cwd=project_root,
                    env=server_env,
                    stdout=server_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            print(
                f"Waiting for server on port {args.port} "
                f"(timeout={args.server_ready_timeout}s)..."
            )
            if not _wait_for_port(args.port, args.server_ready_timeout):
                raise SystemExit(
                    f"Server did not become ready. Check {server_log}"
                )
            print("Server ready.")

            print("\nRunning mini-swe-agent...")
            with mini_log.open("w", encoding="utf-8") as mini_fh:
                mini_result = subprocess.run(
                    mini_cmd,
                    cwd=project_root,
                    env=mini_env,
                    stdout=mini_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

        if mini_result.returncode != 0:
            raise SystemExit(
                f"mini-swe-agent exited with code {mini_result.returncode}. "
                f"Check {mini_log}"
            )

    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=10)

    preds_path = output_dir / "preds.json"
    micro_metrics_path = micro_metrics_dir / "micro_metrics.jsonl"

    print("\nDone.")
    print(f"Predictions    : {preds_path}")
    print(f"Micro metrics  : {micro_metrics_path}")
    print(f"Server log     : {server_log}")
    print(f"mini log       : {mini_log}")

    if not preds_path.exists():
        print("WARNING: preds.json was not created.", file=sys.stderr)
    if not micro_metrics_path.exists():
        print(
            "WARNING: micro_metrics.jsonl was not created. "
            "Check server.log to confirm sparse requests ran.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
