#!/usr/bin/env python3
"""Run AIME benchmarks against an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request

# Allow `python scripts/run_aime_vllm.py` from repo root without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark import create_benchmark_instance  # noqa: E402
from benchmark.utils import make_serializable, save_dataframe_to_csv  # noqa: E402
from sparse_attention_hub.adapters.base import Request, RequestResponse  # noqa: E402
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger  # noqa: E402

MicroMetricLogger.register_metric("vllm_client_request_latency_s", float)
MicroMetricLogger.register_metric("vllm_client_response_chars", int)


def _merge_vllm_run_metadata_into_outputs(
    out_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    """Write ``sparsity_meta.json`` and merge the same keys into ``config.json`` if present."""
    if not metadata:
        return
    sparsity_path = out_dir / "sparsity_meta.json"
    with open(sparsity_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    cfg_path = out_dir / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(metadata)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def _build_vllm_run_metadata(
    attention_config: Optional[Dict[str, Any]],
    base_url: str,
    record_gpu_id: Optional[str],
) -> Dict[str, Any]:
    """Fields intended to match the ``vllm serve`` process this client used."""
    meta: Dict[str, Any] = {
        "vllm_client_base_url": base_url.rstrip("/"),
        "vllm_serve_attention_config": attention_config,
    }
    if record_gpu_id is not None:
        meta["expected_server_cuda_visible_devices"] = record_gpu_id
    if attention_config is not None:
        tk = attention_config.get("topk")
        if tk is not None:
            try:
                r = float(tk)
                meta["sparse_topk_ratio"] = r
                meta["sparse_topk_percent_estimate"] = r * 100.0
            except (TypeError, ValueError):
                pass
    return meta


class VllmChatAdapter:
    """Adapter implementing the benchmark `process_request` interface via chat/completions."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout_s: int = 600,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _chat_completion(self, prompt: str, generation_kwargs: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(generation_kwargs.get("max_new_tokens", 4096)),
            "temperature": float(generation_kwargs.get("temperature", 0.0)),
            "top_p": float(generation_kwargs.get("top_p", 1.0)),
        }
        if "do_sample" in generation_kwargs and not generation_kwargs["do_sample"]:
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach vLLM endpoint: {e}") from e

        parsed = json.loads(raw)
        try:
            return parsed["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected vLLM response format: {parsed}") from e

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
    ) -> RequestResponse:
        del request_kwargs  # Not used for server-backed path.
        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        responses: List[str] = []
        for question in questions:
            prompt = f"{request.context}\n\n{question}{request.answer_prefix}"
            responses.append(self._chat_completion(prompt, generation_kwargs))

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        return RequestResponse(responses=responses)


def _response_char_count(resp: RequestResponse) -> int:
    if isinstance(resp.responses, list):
        return sum(len(str(x)) for x in resp.responses)
    return len(str(resp.responses))


class TimedVllmChatAdapter:
    """Wraps ``VllmChatAdapter`` and logs per-request client micro-metrics."""

    def __init__(
        self,
        inner: VllmChatAdapter,
        metric_logger: Optional[MicroMetricLogger],
        enabled: bool,
    ) -> None:
        self.inner = inner
        self.metric_logger = metric_logger
        self.enabled = enabled
        self._call_idx = 0

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
    ) -> RequestResponse:
        t0 = time.perf_counter()
        out = self.inner.process_request(request, generation_kwargs, request_kwargs)
        if self.enabled and self.metric_logger is not None and self.metric_logger.log_path:
            dt = time.perf_counter() - t0
            meta = {"row_index": self._call_idx, "runner": "run_aime_vllm"}
            self.metric_logger.log("vllm_client_request_latency_s", dt, meta)
            self.metric_logger.log(
                "vllm_client_response_chars",
                _response_char_count(out),
                meta,
            )
            self._call_idx += 1
        return out


def _configure_micro_metrics(micro_dir: Path) -> MicroMetricLogger:
    micro_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=str(micro_dir),
        enabled_metrics="all",
    )
    return metric_logger


def _assign_prediction(
    results_df: Any,
    idx: Any,
    response: RequestResponse,
    questions: Any,
) -> None:
    qlist: List[Any] = questions if isinstance(questions, list) else [questions]
    if isinstance(response.responses, list):
        results_df.at[idx, "predicted_answer"] = response.responses
    else:
        results_df.at[idx, "predicted_answer"] = (
            [response.responses] * len(qlist) if len(qlist) > 1 else response.responses
        )


def _process_vllm_parallel(
    adapter: VllmChatAdapter,
    benchmark: Any,
    dataset_df: Any,
    generation_kwargs: Dict[str, Any],
    request_kwargs: Dict[str, Any],
    concurrency: int,
    metric_logger: Optional[MicroMetricLogger] = None,
    micro_log: bool = False,
) -> Any:
    """Overlap HTTP completions so vLLM can batch multiple in-flight sequences."""
    from tqdm import tqdm

    results_df = dataset_df.copy()
    results_df["predicted_answer"] = None

    work: List[Tuple[Any, Dict[str, Any], Request]] = []
    for idx, row in results_df.iterrows():
        row_max_new = int(row["max_new_tokens"])
        param_max = generation_kwargs.get("max_new_tokens", sys.maxsize)
        max_nt = min(int(param_max), row_max_new)
        gk = {**generation_kwargs, "max_new_tokens": max_nt}
        req = Request(
            context=row["context"],
            questions=row["question"],
            answer_prefix=row.get("answer_prefix", "") or "",
        )
        work.append((idx, gk, req))

    def _worker(
        row_index: Any,
        gk: Dict[str, Any],
        req: Request,
    ) -> Tuple[Any, RequestResponse, float]:
        t0 = time.perf_counter()
        resp = adapter.process_request(req, gk, request_kwargs)
        dt = time.perf_counter() - t0
        return row_index, resp, dt

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(_worker, idx, gk, req): idx for idx, gk, req in work
        }
        for fut in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="vLLM parallel requests",
        ):
            idx = future_to_idx[fut]
            row = results_df.loc[idx]
            _, resp, dt = fut.result()
            if micro_log and metric_logger is not None and metric_logger.log_path:
                meta = {"df_index": repr(idx), "runner": "run_aime_vllm"}
                metric_logger.log("vllm_client_request_latency_s", dt, meta)
                metric_logger.log(
                    "vllm_client_response_chars",
                    _response_char_count(resp),
                    meta,
                )
            _assign_prediction(results_df, idx, resp, row["question"])

    return results_df


def _save_run_artifacts(
    benchmark: Any,
    adapter: Any,
    results_df: Any,
    result_dir: str,
    generation_kwargs: Dict[str, Any],
    request_kwargs: Dict[str, Any],
    extra_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    raw_results_path = result_path / "raw_results.csv"
    save_dataframe_to_csv(results_df, str(raw_results_path), index=False)
    print(f"Saved raw results to {raw_results_path}")

    print("Computing evaluation metrics...")
    metrics: Dict[str, Any] = benchmark.post_run_evaluate(results_df)

    metrics_path = result_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    config_data: Dict[str, Any] = {
        "model_kwargs": make_serializable(getattr(adapter, "model_kwargs", {})),
        "tokenizer_kwargs": make_serializable(getattr(adapter, "tokenizer_kwargs", {})),
        "sparse_attention_config": make_serializable(
            getattr(adapter, "sparse_attention_config", None)
        ),
        "generation_kwargs": make_serializable(generation_kwargs),
        "request_kwargs": make_serializable(request_kwargs),
        "benchmark_name": benchmark.benchmark_name,
        "subsets_to_run": benchmark.subsets_to_run,
        "huggingface_dataset_id": getattr(benchmark, "huggingface_dataset_id", None),
        "runner": "run_aime_vllm.py",
    }
    if extra_config:
        config_data.update(extra_config)

    config_path = result_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved configuration to {config_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIME benchmark via vLLM OpenAI endpoint")
    parser.add_argument("--dataset", choices=["aime2024", "aime2025", "aime2026"], default="aime2025")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:4000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-context-length", type=int, default=65536)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of in-flight /v1/chat/completions calls. Values >1 let vLLM continuous "
            "batching help throughput on one GPU (try 8–16 on B200; tune with vLLM "
            "--max-num-seqs / memory)."
        ),
    )
    parser.add_argument("--result-dir", default="results/aime_vllm")
    parser.add_argument(
        "--micro-metrics",
        action="store_true",
        help=(
            "Write micro_metrics/micro_metrics.jsonl with client-side stats "
            "(request latency, response length). Server-side sparse kernel metrics "
            "require vLLM launched with SAH_METRICS_LOG_DIR if your build supports it."
        ),
    )
    parser.add_argument(
        "--micro-metrics-dir",
        default=None,
        help="Directory for micro_metrics.jsonl (default: <out_dir>/micro_metrics).",
    )
    parser.add_argument(
        "--record-attention-config",
        default=None,
        metavar="JSON",
        help=(
            "JSON string matching the vLLM server's --attention-config; written to "
            "sparsity_meta.json and merged into config.json. "
            "Example: '{\"backend\":\"FLASHINFER_SPARSE\",\"topk\":0.2,\"channel_num\":-1}'"
        ),
    )
    parser.add_argument(
        "--record-gpu-id",
        default=None,
        metavar="ID",
        help=(
            "GPU index used for the vLLM server this client calls (for logging only; "
            "should match CUDA_VISIBLE_DEVICES on the server process)."
        ),
    )
    args = parser.parse_args()

    attn_cfg: Optional[Dict[str, Any]] = None
    if args.record_attention_config is not None and str(args.record_attention_config).strip():
        try:
            parsed = json.loads(args.record_attention_config)
        except json.JSONDecodeError as e:
            print(
                f"error: --record-attention-config must be valid JSON: {e}",
                file=sys.stderr,
            )
            raise SystemExit(2) from e
        if not isinstance(parsed, dict):
            print(
                "error: --record-attention-config must be a JSON object at the top level.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        attn_cfg = parsed

    record_meta = _build_vllm_run_metadata(
        attn_cfg, args.base_url, args.record_gpu_id
    )

    base_adapter = VllmChatAdapter(
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
    )
    benchmark = create_benchmark_instance(args.dataset, [args.dataset])

    request_kwargs: Dict[str, Any] = {"max_context_length": args.max_context_length}
    if args.max_requests is not None:
        request_kwargs["max_requests"] = args.max_requests

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    out_dir = Path(args.result_dir) / args.dataset / "dense_vllm"
    out_dir.mkdir(parents=True, exist_ok=True)

    micro_dir = (
        Path(args.micro_metrics_dir).resolve()
        if args.micro_metrics_dir is not None
        else out_dir / "micro_metrics"
    )
    metric_logger: Optional[MicroMetricLogger] = None
    if args.micro_metrics:
        metric_logger = _configure_micro_metrics(micro_dir)

    adapter: Any = (
        TimedVllmChatAdapter(base_adapter, metric_logger, True)
        if args.micro_metrics
        else base_adapter
    )

    if args.concurrency <= 1:
        metrics = benchmark.run_benchmark(
            adapter=adapter,
            result_dir=str(out_dir),
            generation_kwargs=generation_kwargs,
            request_kwargs=request_kwargs,
        )
        if args.micro_metrics and metric_logger is not None:
            metric_logger.flush()
    else:
        print(f"Loading {benchmark.benchmark_name} datasets: {benchmark.subsets_to_run}")
        dataset_df = benchmark._load_datasets()
        if args.max_requests is not None:
            dataset_df = dataset_df.head(args.max_requests)
        print(f"Loaded {len(dataset_df)} samples")
        benchmark._validate_dataset_size(dataset_df)
        print(
            f"Processing requests through adapter (concurrency={args.concurrency})..."
        )
        results_df = _process_vllm_parallel(
            base_adapter,
            benchmark,
            dataset_df,
            generation_kwargs,
            request_kwargs,
            args.concurrency,
            metric_logger=metric_logger,
            micro_log=bool(args.micro_metrics),
        )
        extra: Dict[str, Any] = {"concurrency": args.concurrency, **record_meta}
        if args.micro_metrics:
            extra["micro_metrics_dir"] = str(micro_dir)
        metrics = _save_run_artifacts(
            benchmark,
            adapter,
            results_df,
            str(out_dir),
            generation_kwargs,
            request_kwargs,
            extra_config=extra,
        )
        if args.micro_metrics and metric_logger is not None:
            metric_logger.flush()

    _merge_vllm_run_metadata_into_outputs(out_dir, record_meta)

    print("\n=== Final Metrics ===")
    print(metrics)
    print(f"\nSaved results to: {out_dir}")
    print(f"Saved sparsity metadata to: {out_dir / 'sparsity_meta.json'}")
    if args.micro_metrics:
        print(f"Saved micro metrics to: {micro_dir / 'micro_metrics.jsonl'}")


if __name__ == "__main__":
    main()
