# /// script
# dependencies = [
#     "marimo",
#     "transformers==5.17.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def kvq_imports():
    import importlib.util
    import inspect
    import json
    import subprocess
    import sys
    import time
    import xml.etree.ElementTree as ET
    from os import environ as ENV
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    return (
        ENV,
        ET,
        Path,
        importlib,
        inspect,
        mo,
        plt,
        subprocess,
        sys,
        time,
        torch,
    )


@app.cell(hide_code=True)
def kvq_title(mo):
    mo.md(r"""
    # KV cache quantization tests (NVFP4 + KIVI)

    This notebook clones skylight-research, applies my kv_quantization changes and runs all the tests on the GPU.

    1. Smoke test
    2. Unit tests
    3. NVFP4 numerics
    4. Cross-device check
    5. Simulator vs real NVFP4
    6. Adapter + greedy generation
    7. WikiText-2 perplexity + KIVI layouts

    Everything runs when the notebook opens. Sections 6 and 7 take longer: set RUN_S6 / RUN_S7 = False in their first cell to skip them. Settings are plain variables in each section's first cell. The model for 6 and 7 is set with `MODEL_NAME` in the model cell.
    """)
    return


@app.cell
def kvq_setup(ENV, Path, importlib, mo, subprocess, sys, torch):
    # clone my fork (or update the existing clone) and use its kv-quantization branch
    KVQ_DIR = Path(ENV.get("KVQ_DIR", "/marimo/notebooks"))        # results/ goes here
    RESULTS_DIR = KVQ_DIR / "results"
    REPO = Path(ENV.get("KVQ_REPO_DIR", Path.home() / "kvq-work" / "skylight-research"))
    _REPO_URL = ENV.get("KVQ_REPO_URL", "https://github.com/mangeshpoojan/skylight-research.git")
    _BRANCH = ENV.get("KVQ_BRANCH", "kv-quantization")

    def _git(*args, cwd=REPO):
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{done.stderr}")
        return done.stdout.strip()

    _log = []
    if not (REPO / ".git").exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--branch", _BRANCH, _REPO_URL, str(REPO), cwd=REPO.parent)
        _log.append(f"cloned `{_REPO_URL}` branch `{_BRANCH}`")
    else:
        # an older clone may point somewhere else: switch it to the fork and pull the latest commit
        _git("remote", "set-url", "origin", _REPO_URL)
        _git("fetch", "--quiet", "origin", _BRANCH)
        _git("checkout", "--quiet", "-B", _BRANCH, f"origin/{_BRANCH}")
        _log.append(f"updated the existing clone to `{_REPO_URL}` branch `{_BRANCH}`")
    if not (REPO / "sparse_attention_hub" / "kv_quantization").exists():
        raise RuntimeError(f"branch {_BRANCH!r} of {_REPO_URL} has no sparse_attention_hub/kv_quantization")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # use the clone

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HAS_MODELOPT = importlib.util.find_spec("modelopt") is not None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mo.md(
        "**Setup**\n\n" + "\n".join(f"- {line}" for line in _log)
        + f"\n- repo: `{REPO}` @ `{_git('rev-parse', '--short', 'HEAD')}` ({_git('log', '-1', '--format=%s')})"
        + f"\n- device: `{DEVICE}` ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else DEVICE}), torch `{torch.__version__}`"
        + f"\n- nvidia-modelopt: {'installed' if HAS_MODELOPT else 'missing (real nvfp4 parts are skipped)'}"
    )
    return DEVICE, HAS_MODELOPT, REPO


@app.cell
def kvq_package(Path, REPO):
    import sparse_attention_hub.kv_quantization as kvq
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM

    from sparse_attention_hub.kv_quantization.constants import E4M3_MAX, E4M3_MIN_SUBNORMAL
    from sparse_attention_hub.kv_quantization.evaluation import perplexity, wikitext2_test_ids
    from sparse_attention_hub.kv_quantization.nvfp4 import (
        bits_per_element,
        fake_quantize_nvfp4,
        q_e2m1,
        q_e4m3,
        quantize_nvfp4,
    )

    # make sure it's importing from the clone
    assert Path(kvq.__file__).resolve().is_relative_to(REPO.resolve()), f"imported {kvq.__file__}, not the clone"
    E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DynamicCache,
        E2M1_GRID,
        E4M3_MAX,
        E4M3_MIN_SUBNORMAL,
        Qwen3Config,
        Qwen3ForCausalLM,
        bits_per_element,
        fake_quantize_nvfp4,
        kvq,
        perplexity,
        q_e2m1,
        q_e4m3,
        quantize_nvfp4,
        wikitext2_test_ids,
    )


@app.cell
def kvq_summary_helper(mo):
    def kvq_summary(title, checks):
        failed = [c["check"] for c in checks if not c["ok"]]
        return mo.vstack([
            mo.md(f"### {title}: summary"),
            mo.callout(
                mo.md(f"**{len(checks) - len(failed)}/{len(checks)} checks passed**" + (f": failed {failed}" if failed else "")),
                kind="danger" if failed else "success",
            ),
            mo.ui.table([{**c, "ok": "PASS" if c["ok"] else "FAIL"} for c in checks], selection=None),
        ])

    return (kvq_summary,)


@app.cell(hide_code=True)
def s1_title(mo):
    mo.md(r"""
    ---
    ## 1. Smoke test

    Quick check that everything is wired up: the package imports from the clone, the `quantize_kv_cache` flag only accepts the two backends, `fake_quantize_nvfp4` gives values on the NVFP4 grid, both caches work on a small random Qwen3 (per token and KIVI layout), and `ModelAdapterHF` creates the cache in both prefill paths.
    """)
    return


@app.cell
def s1_versions(kvq, mo):
    import importlib.metadata as _metadata

    def _version(name):
        try:
            return _metadata.version(name)
        except _metadata.PackageNotFoundError:
            return "not installed"

    mo.hstack([
        mo.ui.table([{"package": n, "version": _version(n)} for n in ("torch", "transformers", "nvidia-modelopt", "datasets")], selection=None),
        mo.ui.table([{"backend": k, "what it is": v} for k, v in kvq.describe_backends().items()], selection=None),
    ])
    return


@app.cell
def s1_flags(kvq):
    # flag should accept the backend names or False, anything else should raise
    _accepted = [kvq.validate_backend(name) for name in kvq.BACKENDS]
    _rejected = []
    for _bad in (True, "nvfp44", "int4"):
        try:
            kvq.validate_backend(_bad)
        except ValueError:
            _rejected.append(repr(_bad))
    check_flags = {
        "check": "flag validation",
        "ok": _accepted == list(kvq.BACKENDS) and len(_rejected) == 3,
        "detail": f"accepted {_accepted}, rejected {', '.join(_rejected)}",
    }
    check_flags
    return (check_flags,)


@app.cell
def s1_round_trip(
    DEVICE,
    E2M1_GRID,
    bits_per_element,
    fake_quantize_nvfp4,
    quantize_nvfp4,
    torch,
):
    # quantize some random values and check the output
    _x = torch.randn(8, 256, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    _y = fake_quantize_nvfp4(_x)
    _codes, _block_scale, _tensor_scale = quantize_nvfp4(_x)
    _grid = torch.tensor(E2M1_GRID, device=DEVICE)
    _on_grid = bool((_codes.abs().unsqueeze(-1) == _grid).any(-1).all())
    _rebuilt = (_codes * (_block_scale * _tensor_scale)).reshape(_x.shape)
    _rel = ((_y - _x).norm() / _x.norm()).item()
    check_round_trip = {
        "check": "fake_quantize_nvfp4 round trip",
        "ok": _on_grid and torch.equal(_rebuilt, _y) and _y.shape == _x.shape and _y.dtype == _x.dtype and _rel < 0.15,
        "detail": f"codes on E2M1 grid: {_on_grid}; codes*scales == output: {torch.equal(_rebuilt, _y)}; "
        f"relative error {_rel:.4f}; {bits_per_element(_x.numel()):.3f} bits/element if packed",
    }
    check_round_trip
    return (check_round_trip,)


@app.cell
def s1_round_trip_real(DEVICE, HAS_MODELOPT, fake_quantize_nvfp4, mo, torch):
    # same values as the round trip above, now through the real NVFP4 quantizer (modelopt)
    _x = torch.randn(8, 256, generator=torch.Generator().manual_seed(0)).to(DEVICE)

    if DEVICE != "cuda" or not HAS_MODELOPT:
        check_round_trip_real = {"check": "fake == real NVFP4 round trip", "ok": True, "detail": "skipped, needs CUDA + modelopt"}
        _out = mo.md("Skipped: the real NVFP4 quantizer needs CUDA and modelopt.")
    else:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor as _NVFP4QTensor

        # real: pack to 4-bit codes + scales, then unpack
        _packed, _scale, _scale2 = _NVFP4QTensor.quantize(_x, 16)
        _real = _packed.dequantize(dtype=torch.float32, scale=_scale, double_scale=_scale2, block_sizes={-1: 16})
        # fake: the simulator
        _fake = fake_quantize_nvfp4(_x)

        def _rel(y):
            return ((y - _x).norm() / _x.norm()).item()

        _same = torch.equal(_fake, _real)
        check_round_trip_real = {
            "check": "fake == real NVFP4 round trip",
            "ok": _same,
            "detail": f"identical to real: {_same} ({int((_fake != _real).sum())} of {_x.numel()} values differ); "
            f"relative error fake {_rel(_fake):.4f} vs real {_rel(_real):.4f}",
        }
        _rows = [
            {
                "x": round(_x[0, i].item(), 5),
                "real nvfp4": _real[0, i].item(),
                "fake nvfp4": _fake[0, i].item(),
                "real == fake": _real[0, i].item() == _fake[0, i].item(),
            }
            for i in range(16)
        ]
        _summary = [
            {"output": "real nvfp4", "relative error": round(_rel(_real), 6), "values different from real": 0},
            {"output": "fake nvfp4", "relative error": round(_rel(_fake), 6), "values different from real": int((_fake != _real).sum())},
        ]
        _out = mo.vstack([
            mo.md("Same 2048 values through real NVFP4 (modelopt) and the simulator. First block of 16:"),
            mo.ui.table(_rows, selection=None),
            mo.ui.table(_summary, selection=None),
        ])
    _out
    return (check_round_trip_real,)


@app.cell
def s1_caches(
    DEVICE,
    DynamicCache,
    HAS_MODELOPT,
    Qwen3Config,
    Qwen3ForCausalLM,
    kvq,
    mo,
    torch,
):
    # small random Qwen3 so nothing needs to be downloaded
    _config = Qwen3Config(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    _model = Qwen3ForCausalLM(_config).to(DEVICE).eval()
    _prompt = torch.randint(0, 256, (1, 200), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    _steps = 40

    def _decode(cache):
        tokens = []
        with torch.no_grad():
            out = _model(_prompt, past_key_values=cache, use_cache=True)
            for _ in range(_steps):
                token = out.logits[:, -1:].argmax(-1)
                tokens.append(int(token))
                out = _model(token, past_key_values=out.past_key_values, use_cache=True)
        return tokens, bool(torch.isfinite(out.logits).all()), cache.get_seq_length()

    _reference, _, _ = _decode(DynamicCache(config=_config))
    _rows = []
    for _backend in ["fake_nvfp4"] + (["nvfp4"] if DEVICE == "cuda" and HAS_MODELOPT else []):
        for _axis_key, _layout in ((-1, "per token"), (0, "KIVI: keys per channel")):
            # residual_length 32 so most of the 200 prompt tokens get quantized
            _cache = kvq.create_quantized_kv_cache(_config, _backend, residual_length=32, axis_key=_axis_key)
            _tokens, _finite, _length = _decode(_cache)
            _rows.append({
                "backend": _backend, "key layout": _layout, "finite logits": _finite, "cached tokens": _length,
                "tokens equal to exact cache": f"{sum(a == b for a, b in zip(_tokens, _reference))}/{_steps}",
            })
    check_caches = {
        "check": "quantized caches decode",
        "ok": all(r["finite logits"] and r["cached tokens"] == 200 + _steps for r in _rows),
        "detail": f"{len(_rows)} backend/layout combinations, {_steps} decode steps each",
    }
    mo.vstack([
        mo.md("Random weights so the token match doesn't mean much, just checking that every cache runs."),
        mo.ui.table(_rows, selection=None),
    ])
    return (check_caches,)


@app.cell
def s1_adapter(inspect):
    from sparse_attention_hub.adapters.huggingface import ModelAdapterHF

    _params = inspect.signature(ModelAdapterHF.__init__).parameters
    _calls = inspect.getsource(ModelAdapterHF.process_request).count("past_key_values=self._create_kv_cache()")
    check_adapter_wiring = {
        "check": "adapter integration",
        "ok": "quantize_kv_cache" in _params and "kv_quantization_kwargs" in _params and _calls == 2,
        "detail": f"flag parameters present; _create_kv_cache() used in {_calls}/2 prefill calls (dense and sparse)",
    }
    check_adapter_wiring
    return ModelAdapterHF, check_adapter_wiring


@app.cell(hide_code=True)
def s1_summary(
    check_adapter_wiring,
    check_caches,
    check_flags,
    check_round_trip,
    check_round_trip_real,
    kvq_summary,
):
    kvq_summary("1. Smoke test", [check_flags, check_round_trip, check_round_trip_real, check_caches, check_adapter_wiring])
    return


@app.cell(hide_code=True)
def s2_title(mo):
    mo.md(r"""
    ---
    ## 2. Unit tests

    Runs the pytest tests in `tests/unit/kv_quantization`.
    """)
    return


@app.cell
def s2_pytest(ENV, ET, REPO, mo, subprocess, sys):
    _tests = REPO / "tests" / "unit" / "kv_quantization"
    _xml = REPO.parent / "kvq_unit_tests.xml"
    with mo.status.spinner("running pytest ..."):
        unit_pytest_run = subprocess.run(
            [sys.executable, "-m", "pytest", str(_tests), "-p", "no:cacheprovider", "-rs", f"--junitxml={_xml}"],
            cwd=REPO, capture_output=True, text=True, env={**ENV, "PYTHONPATH": str(REPO)},
        )
    unit_test_rows = []
    for _case in ET.parse(_xml).getroot().iter("testcase"):
        _status, _message = "passed", ""
        for _tag in ("failure", "error", "skipped"):
            _node = _case.find(_tag)
            if _node is not None:
                _status = {"failure": "FAILED", "error": "ERROR", "skipped": "skipped"}[_tag]
                _message = (_node.get("message") or "")[:200]
        unit_test_rows.append({
            "file": _case.get("classname", "").split(".tests.")[-1].split(".")[0], "test": _case.get("name"),
            "status": _status, "seconds": round(float(_case.get("time", 0)), 3), "message": _message,
        })
    _by_file = {}
    for _row in unit_test_rows:
        _counts = _by_file.setdefault(_row["file"], {"file": _row["file"], "passed": 0, "skipped": 0, "FAILED": 0, "ERROR": 0})
        _counts[_row["status"]] += 1
    _bad = [r["test"] for r in unit_test_rows if r["status"] in ("FAILED", "ERROR")]
    check_unit_tests = {
        "check": "unit tests",
        "ok": unit_pytest_run.returncode == 0 and not _bad and len(unit_test_rows) > 0,
        "detail": f"{sum(r['status'] == 'passed' for r in unit_test_rows)} passed, "
        f"{sum(r['status'] == 'skipped' for r in unit_test_rows)} skipped, failed: {_bad or 'none'}",
    }
    mo.vstack([
        mo.md(f"`pytest` exited with code **{unit_pytest_run.returncode}**"),
        mo.ui.table(list(_by_file.values()), selection=None),
        mo.accordion({
            "every test": mo.ui.table(unit_test_rows, selection=None, page_size=25),
            "pytest output (last 150 lines)": mo.plain_text("\n".join(unit_pytest_run.stdout.splitlines()[-150:]) + unit_pytest_run.stderr),
        }),
    ])
    return (check_unit_tests,)


@app.cell(hide_code=True)
def s2_summary(check_unit_tests, kvq_summary):
    kvq_summary("2. Unit tests", [check_unit_tests])
    return


@app.cell(hide_code=True)
def s3_title(mo):
    mo.md(r"""
    ---
    ## 3. NVFP4 numerics

    NVFP4 stores each value as `code * block_scale * tensor_scale`:
    - code: 4-bit E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and the negatives)
    - block_scale: 8-bit E4M3, one per 16 values
    - tensor_scale: one FP32 value for the whole tensor
    """)
    return


@app.cell
def s3_e2m1(DEVICE, mo, q_e2m1, torch):
    # midpoints are ties -> round half to even
    _cases = {
        0.2: 0.0, 0.25: 0.0, 0.3: 0.5, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0,
        3.5: 4.0, 5.0: 4.0, 5.1: 6.0, 7.0: 6.0, 1e9: 6.0, -2.5: -2.0, -5.0: -4.0,
    }
    _got = q_e2m1(torch.tensor(list(_cases), device=DEVICE)).tolist()
    _rows = [{"input": x, "expected": e, "q_e2m1": g, "ok": e == g} for (x, e), g in zip(_cases.items(), _got)]
    check_e2m1 = {"check": "E2M1 rounding and ties", "ok": all(r["ok"] for r in _rows), "detail": "round half to even, saturate at 6"}
    mo.vstack([mo.md("#### E2M1 (the 4-bit element)"), mo.ui.table(_rows, selection=None)])
    return (check_e2m1,)


@app.cell
def s3_e4m3(E4M3_MAX, mo, q_e4m3, torch):
    # compare q_e4m3 with torch's float8_e4m3fn on all bf16 values
    _bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    _values = _bits.view(torch.bfloat16).float()
    _values = _values[torch.isfinite(_values) & (_values.abs() <= E4M3_MAX)]
    _mismatches = int((q_e4m3(_values) != _values.to(torch.float8_e4m3fn).float()).sum())
    check_e4m3 = {
        "check": "E4M3 == torch.float8_e4m3fn",
        "ok": _mismatches == 0,
        "detail": f"{_values.numel():,} bf16 values, {_mismatches} mismatches (CPU)",
    }
    mo.md(f"#### E4M3 (the 8-bit block scale)\n\nAll {_values.numel():,} finite bf16 values in [-448, 448]: **{_mismatches} mismatches** against `torch.float8_e4m3fn`.")
    return (check_e4m3,)


@app.cell
def s3_anatomy(DEVICE, mo, quantize_nvfp4, torch):
    # two blocks of 16, showing codes and scales
    _x = torch.tensor(
        [0.02, -0.4, 1.3, 0.9, -2.2, 0.05, 0.6, -0.7, 1.9, 0.1, -0.3, 0.8, 0.25, -1.1, 0.45, 0.33]
        + [8.0, -3.0, 0.5, 2.0, 7.9, -0.02, 1.0, 4.4, -6.5, 3.3, 0.9, -1.2, 5.5, 2.2, -0.1, 0.7],
        device=DEVICE,
    )
    _codes, _block_scale, _tensor_scale = quantize_nvfp4(_x)
    _dequant = (_codes * (_block_scale * _tensor_scale)).flatten()
    mo.vstack([
        mo.md(
            "#### Two blocks in detail\n\n"
            f"`tensor_scale = amax / (448*6) = {_x.abs().max().item()} / 2688 = {_tensor_scale.item():.6g}`. "
            "Block scale = `block_amax / 6 / tensor_scale` rounded to E4M3, code = "
            "`x / (block_scale * tensor_scale)` rounded to E2M1."
        ),
        mo.ui.table([
            {"block": i // 16, "x": round(_x[i].item(), 4), "code (E2M1)": _codes.flatten()[i].item(),
             "block_scale (E4M3)": _block_scale.flatten()[i // 16].item(), "dequantized": round(_dequant[i].item(), 4)}
            for i in range(32)
        ], selection=None, page_size=32),
    ])
    return


@app.cell
def s3_staircase(fake_quantize_nvfp4, plt, torch):
    # each row is a block with max 7 so the scale stays the same, first element shows the rounding
    _x = torch.linspace(-7, 7, 2001)
    _blocks = torch.zeros(2001, 16)
    _blocks[:, 0] = _x
    _blocks[:, 1] = 7.0
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_x, _x, lw=0.8, color="grey", label="input")
    _ax.plot(_x, fake_quantize_nvfp4(_blocks)[:, 0], lw=1.4, label="fake_quantize_nvfp4 (block max 7)")
    _ax.set_xlabel("x")
    _ax.legend()
    _ax.set_title("fake_quantize_nvfp4 on a ramp")
    _fig
    return


@app.cell
def s3_conventions(DEVICE, E4M3_MIN_SUBNORMAL, mo, quantize_nvfp4, torch):
    # tiny block next to a huge one: its scale hits the 2^-9 floor
    _x = torch.cat([torch.full((16,), 1e4), torch.linspace(1e-3, 2e-3, 16)]).to(DEVICE)
    _codes, _block_scale, _tensor_scale = quantize_nvfp4(_x)
    _small = (_codes * (_block_scale * _tensor_scale))[1]
    _huge = _codes[0] * (_block_scale[0] * _tensor_scale)
    _rows = [
        {"which block": "huge (1e4)", "block scale": float(_block_scale[0]), "relative error": round(float((_huge - _x[:16]).norm() / _x[:16].norm()), 4)},
        {"which block": "tiny (1e-3)", "block scale": float(_block_scale[1]), "relative error": round(float((_small - _x[16:]).norm() / _x[16:].norm()), 4)},
    ]
    _table = "| block | block scale | relative error |\n|---|---|---|\n" + "".join(
        f"| {r['which block']} | {r['block scale']:.9g} | {r['relative error']} |\n" for r in _rows
    )
    mo.md(
        "#### Tiny blocks and the 2^-9 floor\n\nThe block scale is block_amax / (6 * tensor_scale), clamped to [2^-9, 448] "
        f"(2^-9 = {E4M3_MIN_SUBNORMAL} is the smallest E4M3 subnormal, same rule as modelopt). A block that is tiny "
        "compared to the tensor max ends up at the floor and mostly rounds to zero:\n\n" + _table
    )
    return


@app.cell
def s3_numel():
    NUMERICS_NUMEL = 1 << 20  # elements per distribution in the invariants check below
    return (NUMERICS_NUMEL,)


@app.cell
def s3_invariants(
    DEVICE,
    E2M1_GRID,
    NUMERICS_NUMEL,
    fake_quantize_nvfp4,
    mo,
    quantize_nvfp4,
    torch,
):
    _g = torch.Generator().manual_seed(0)
    _shape = (NUMERICS_NUMEL // 256, 256)
    _u = lambda: torch.rand(_shape, generator=_g).clamp(1e-7, 1 - 1e-7)  # noqa: E731
    _corpus = {
        "gaussian": torch.randn(_shape, generator=_g),
        "laplace": -torch.sign(_u() - 0.5) * torch.log1p(-2 * (_u() - 0.5).abs()),
        "student_t (df 2)": torch.randn(_shape, generator=_g) / torch.sqrt(-torch.log(_u())),
        "sparse (99% zero)": torch.randn(_shape, generator=_g) * (_u() < 0.01),
        "KV-like outlier channels": torch.randn(_shape, generator=_g) * 0.5 + torch.zeros(256).index_fill(0, torch.tensor([3, 17, 40, 200]), 40.0),
        "wide range rows": torch.randn(_shape, generator=_g) * torch.exp2(torch.randint(-20, 20, (_shape[0], 1), generator=_g).float()),
    }
    _grid = torch.tensor(E2M1_GRID, device=DEVICE)
    _rows = []
    for _name, _x in _corpus.items():
        _x = _x.to(DEVICE)
        _codes, _block_scale, _tensor_scale = quantize_nvfp4(_x)
        _y = fake_quantize_nvfp4(_x)
        _step = (_block_scale * _tensor_scale).expand_as(_codes).reshape(_x.shape)
        _normal = (_block_scale >= 2.0**-6).expand_as(_codes).reshape(_x.shape)
        _steps = (_y - _x).abs() / _step
        _rows.append({
            "distribution": _name,
            "relative error": round(((_y - _x).norm() / _x.norm()).item(), 5),
            "max error / step": round(_steps.max().item(), 4),
            "max error / step (normal-scale blocks)": round(_steps[_normal].max().item(), 4) if _normal.any() else 0.0,
            "codes after re-quantize": "same" if torch.equal(quantize_nvfp4(_y)[0], _codes) else "differ",
            "re-quantize change (relative)": f"{((fake_quantize_nvfp4(_y) - _y).norm() / _y.norm()).item():.1e}",
            "sign symmetric": torch.equal(fake_quantize_nvfp4(-_x), -_y),
            "codes on grid": bool((_codes.abs().unsqueeze(-1) == _grid).any(-1).all()),
        })
    check_invariants = {
        "check": "invariants over distributions",
        "ok": all(r["sign symmetric"] and r["codes on grid"] and r["max error / step (normal-scale blocks)"] <= 1.0 + 1e-6
                  and float(r["re-quantize change (relative)"]) < 1e-3 for r in _rows),
        "detail": "sign symmetric, codes on grid, error <= 1 step in normal-scale blocks, re-quantizing moves values < 0.1%",
    }
    mo.vstack([
        mo.md(
            "#### Invariants\n\nmax error / step = error divided by the block step (block_scale * tensor_scale). In blocks with a "
            "normal scale (>= 2^-6) it's always <= 1. With a subnormal scale the scale can round down and clip at 6, so a few "
            "elements can go a bit over (same as real NVFP4). Quantizing the output again keeps the codes, but the tensor "
            "scale can shift by one float32 bit, so values can move by ~1e-7."
        ),
        mo.ui.table(_rows, selection=None, page_size=12),
    ])
    return (check_invariants,)


@app.cell(hide_code=True)
def s3_summary(check_e2m1, check_e4m3, check_invariants, kvq_summary):
    kvq_summary("3. NVFP4 numerics", [check_e2m1, check_e4m3, check_invariants])
    return


@app.cell(hide_code=True)
def s4_title(mo):
    mo.md(r"""
    ---
    ## 4. CPU vs GPU check

    Runs `fake_quantize_nvfp4` on the molab CPU and GPU (13 inputs x 3 dtypes x 2 layouts) and compares the outputs bit by bit. It passes if both run without errors and give the same accuracy. The GPU can differ in the last bit because CUDA divides by a constant using the reciprocal.
    """)
    return


@app.cell
def s4_corpus(mo, torch):
    import hashlib as _hashlib
    import math as _math

    _MASK = 0xFFFFFFFF

    def _hash32(counter):
        # lowbias32 style hash, int64 so it doesn't overflow
        x = counter & _MASK
        x = x ^ (x >> 16)
        x = (x * 0x7FEB352D) & _MASK
        x = x ^ (x >> 15)
        x = (x * 0x5BD1E995) & _MASK
        return x ^ (x >> 16)

    def _uniform(stream, *shape):
        # k / 2^24 is exact in float32
        counter = torch.arange(_math.prod(shape), dtype=torch.int64) + stream * 0x9E3779B1
        return (_hash32(_hash32(counter)) >> 8).to(torch.float32).reshape(shape) / 2**24

    def _normal(stream, *shape):
        # sum of 12 uniforms - 6 (roughly normal)
        total = torch.zeros(shape)
        for k in range(12):
            total = total + _uniform(1000 + stream * 16 + k, *shape)
        return total - 6.0

    def _portable_corpus():
        shape = (512, 256)
        corpus = {
            "gaussian": _normal(1, *shape),
            "uniform": _uniform(2, *shape) * 2 - 1,
            "heavy_tailed": _normal(3, *shape) / (_uniform(4, *shape) + 2.0**-10),
            "sparse": _normal(5, *shape) * (_uniform(6, *shape) < 0.01),
            "constant": torch.full(shape, 0.37),
        }
        huge = _normal(7, *shape)
        huge[0, 0] = 1e6
        corpus["huge_outlier"] = huge
        tiny = _normal(8, *shape) * 1e4
        tiny[:, :16] = _normal(9, 512, 16) * 1e-8
        corpus["tiny_next_to_huge"] = tiny
        exponents = (_hash32(torch.arange(512) + 77777) % 40 - 20).tolist()
        corpus["wide_range_rows"] = _normal(10, *shape) * torch.tensor([[2.0**e] for e in exponents])
        keys = _normal(11, *shape) * 0.5
        keys[:, [3, 17, 40, 200]] += torch.tensor([30.0, -45.0, 60.0, 25.0])
        corpus["kv_like_outlier_channels"] = keys
        midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # E2M1 ties once scaled
        ties = midpoints[_hash32(torch.arange(512 * 256) + 88888).reshape(shape) % 7]
        ties[:, ::16] = 6.0
        corpus["exact_ties"] = ties * torch.where(_uniform(12, *shape) < 0.5, -1.0, 1.0)
        corpus["awkward_len_100"] = _normal(13, 64, 100)
        corpus["awkward_len_4097"] = _normal(14, 3, 4097)
        corpus["bf16_exact_gaussian"] = _normal(15, *shape).to(torch.bfloat16).float()
        return corpus

    def xdev_digest(tensor):
        return _hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

    xdev_corpus = _portable_corpus()
    mo.md(f"{len(xdev_corpus)} test inputs")
    return xdev_corpus, xdev_digest


@app.cell
def s4_run(fake_quantize_nvfp4, mo, torch, xdev_corpus, xdev_digest):
    xdev_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def _run(x, layout):
        if layout == "per_channel":  # per channel = transpose first
            return fake_quantize_nvfp4(x.transpose(-1, -2)).transpose(-1, -2)
        return fake_quantize_nvfp4(x)

    xdev_hashes, xdev_diffs, xdev_errors = {}, [], []
    with mo.status.progress_bar(total=3 * 2 * len(xdev_corpus), title="quantizing") as _bar:
        for _dtype in (torch.float32, torch.bfloat16, torch.float16):
            for _layout in ("per_token", "per_channel"):
                for _name, _x32 in xdev_corpus.items():
                    _bar.update()
                    _x = _x32.to(_dtype)
                    if not torch.isfinite(_x).all():
                        continue  # 1e6 overflows fp16
                    _case = f"{str(_dtype).split('.')[1]}|{_layout}|{_name}"
                    _reference = None
                    for _device in xdev_devices:
                        try:
                            _out = _run(_x.to(_device), _layout).cpu().float()
                        except Exception as _exc:
                            xdev_errors.append({"case": _case, "device": _device, "error": repr(_exc)[:200]})
                            continue
                        xdev_hashes.setdefault(_case, {})[_device] = xdev_digest(_out.to(_dtype))
                        _err = ((_out - _x.float()).norm() / _x.float().norm()).item()
                        if _reference is None:
                            _reference = (_out, _err)
                        elif not torch.equal(_out, _reference[0]):
                            xdev_diffs.append({
                                "case": _case, "device vs cpu": _device,
                                "elements differing": f"{(_out != _reference[0]).float().mean().item():.2e}",
                                "max |diff| / amax": f"{((_out - _reference[0]).abs().max() / _x.float().abs().max()).item():.1e}",
                                "rel. error cpu": round(_reference[1], 6), "rel. error device": round(_err, 6),
                                "_err_gap": abs(_err - _reference[1]) / max(_reference[1], 1e-12),
                                "_finite": bool(torch.isfinite(_out).all()),
                            })
    _identical = sum(len(set(h.values())) == 1 for h in xdev_hashes.values())
    check_xdev_runs = {
        "check": "runs on every device",
        "ok": not xdev_errors and all(len(h) == len(xdev_devices) for h in xdev_hashes.values()),
        "detail": f"devices {xdev_devices}, {len(xdev_hashes)} cases, {len(xdev_errors)} errors",
    }
    check_xdev_accuracy = {
        "check": "same accuracy on every device",
        "ok": all(d["_finite"] and d["_err_gap"] < 1e-3 for d in xdev_diffs),
        "detail": "relative error agrees within 0.1% wherever the bits differ",
    }
    mo.vstack([
        mo.md(f"#### CPU vs GPU\n\n**{_identical}/{len(xdev_hashes)} cases bit-identical across {xdev_devices}.**"),
        mo.ui.table(xdev_errors, selection=None) if xdev_errors else mo.md("No device raised an error."),
        mo.md("Where the bits differ, and by how much:") if xdev_diffs else mo.md(""),
        mo.ui.table([{k: v for k, v in d.items() if not k.startswith("_")} for d in xdev_diffs], selection=None, page_size=10) if xdev_diffs else mo.md(""),
    ])
    return check_xdev_accuracy, check_xdev_runs


@app.cell(hide_code=True)
def s4_summary(check_xdev_accuracy, check_xdev_runs, kvq_summary):
    kvq_summary("4. Cross-device", [check_xdev_runs, check_xdev_accuracy])
    return


@app.cell(hide_code=True)
def s5_title(mo):
    mo.md(r"""
    ---
    ## 5. Simulator vs real NVFP4 (modelopt)

    The simulator should give exactly the same bits as modelopt's `NVFP4QTensor` (which the `nvfp4` backend uses). First on 12 stress distributions, then I run a small random Qwen3 through the real cache and the simulated cache and compare the logits. Needs CUDA and modelopt.
    """)
    return


@app.cell
def s5_numel(DEVICE, HAS_MODELOPT, mo):
    mo.stop(DEVICE != "cuda" or not HAS_MODELOPT, mo.callout(mo.md("section 5 needs CUDA and nvidia-modelopt: skipped."), kind="warn"))
    # reuse the comparison from the heavy test file
    from tests.integration.kv_quantization.test_nvfp4_simulator_heavy import (
        compare_with_modelopt,
        distribution_corpus,
    )

    MODELOPT_NUMEL = 1 << 20  # elements per distribution compared against modelopt
    return MODELOPT_NUMEL, compare_with_modelopt, distribution_corpus


@app.cell
def s5_tensors(MODELOPT_NUMEL, compare_with_modelopt, distribution_corpus, mo):
    _rows = []
    with mo.status.spinner("comparing 12 distributions ..."):
        for _name, _x in distribution_corpus(MODELOPT_NUMEL, seed=1).items():
            _stats = compare_with_modelopt(_x)
            _rows.append({
                "distribution": _name,
                "elements differing": _stats["element_mismatch_rate"],
                "block scales differing": round(_stats["block_scale_mismatch_rate"], 6),
                "unexplained": _stats["unexplained_block_scale_mismatches"] + _stats["unexplained_element_mismatches"],
                "rel. error simulator": round(_stats["relative_error_simulator"], 6),
                "rel. error modelopt": round(_stats["relative_error_modelopt"], 6),
            })
    check_bitexact = {
        "check": "simulator bit-exact with modelopt",
        "ok": all(r["elements differing"] == 0 for r in _rows),
        "detail": f"{len(_rows)} distributions x {MODELOPT_NUMEL:,} elements, differing elements: {sum(r['elements differing'] for r in _rows)}",
    }
    mo.vstack([mo.md("#### Tensors"), mo.ui.table(_rows, selection=None, page_size=24)])
    return (check_bitexact,)


@app.cell
def s5_cache_twin(
    DEVICE,
    HAS_MODELOPT,
    Qwen3Config,
    Qwen3ForCausalLM,
    kvq,
    mo,
    torch,
):
    mo.stop(DEVICE != "cuda" or not HAS_MODELOPT)
    _config = Qwen3Config(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=4, head_dim=32, max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    _model = Qwen3ForCausalLM(_config).to("cuda", torch.bfloat16).eval()
    _prompt = torch.randint(0, 512, (1, 600), generator=torch.Generator().manual_seed(1)).cuda()

    def _logits(backend, axis_key, **extra):
        cache = kvq.create_quantized_kv_cache(_config, backend, axis_key=axis_key, residual_length=16, quantize_prefill=True, **extra)
        with torch.no_grad():
            return _model(_prompt, past_key_values=cache, use_cache=True).logits.float()

    _rows = []
    for _axis_key, _layout in ((-1, "per token"), (0, "KIVI keys per channel")):
        _real = _logits("nvfp4", _axis_key)
        _sim = _logits("fake_nvfp4", _axis_key)
        _rows.append({
            "layout": _layout,
            "logits identical to real nvfp4": torch.equal(_sim, _real),
            "max |logit diff|": (_sim - _real).abs().max().item(),
        })
    check_cache_twin = {
        "check": "simulator cache == real cache",
        "ok": all(r["logits identical to real nvfp4"] for r in _rows),
        "detail": "tiny Qwen3, 600-token prefill with quantized KV, both layouts",
    }
    mo.vstack([mo.md("#### Caches inside a model"), mo.ui.table(_rows, selection=None)])
    return (check_cache_twin,)


@app.cell(hide_code=True)
def s5_summary(check_bitexact, check_cache_twin, kvq_summary):
    kvq_summary("5. Simulator vs modelopt", [check_bitexact, check_cache_twin])
    return


@app.cell(hide_code=True)
def model_title(mo):
    mo.md(r"""
    ---
    ## Model for sections 6 and 7

    Loads `MODEL_NAME` once in bf16. Sections 6 and 7 each have a RUN_S* flag in their first cell.
    """)
    return


@app.cell
def model_load(
    AutoModelForCausalLM,
    AutoTokenizer,
    DEVICE,
    DynamicCache,
    HAS_MODELOPT,
    kvq,
    mo,
    torch,
):
    mo.stop(DEVICE != "cuda", mo.callout(mo.md("sections 6 and 7 need an NVIDIA GPU."), kind="warn"))

    MODEL_NAME = "Qwen/Qwen3-4B"  # change the model here
    with mo.status.spinner(f"loading {MODEL_NAME} ..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda").eval()

    def make_cache_factory(backend, axis_key=-1, axis_value=-1, quantize_prefill=False):
        # returns a function that makes a new cache (need a new one per request)
        if backend == "bf16":
            return lambda: DynamicCache(config=model.config)
        kwargs = {"axis_key": axis_key, "axis_value": axis_value, "residual_length": 128, "quantize_prefill": quantize_prefill}
        return lambda: kvq.create_quantized_kv_cache(model.config, backend, **kwargs)

    # (label, backend, axis_key, axis_value)
    CACHE_VARIANTS = [
        v for v in [
            ("bf16", "bf16", -1, -1),
            ("fake_nvfp4", "fake_nvfp4", -1, -1),
            ("nvfp4 (real)", "nvfp4", -1, -1),
            ("KIVI fake_nvfp4", "fake_nvfp4", 0, -1),
            ("KIVI nvfp4 (real)", "nvfp4", 0, -1),
        ]
        if v[1] != "nvfp4" or HAS_MODELOPT
    ]
    mo.md(f"Loaded `{MODEL_NAME}`: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters, bf16")
    return CACHE_VARIANTS, MODEL_NAME, make_cache_factory, model, tokenizer


@app.cell(hide_code=True)
def s6_title(mo):
    mo.md(r"""
    ---
    ## 6. Adapter + greedy generation

    First one request through `ModelAdapterHF` with every flag and layout. Then 64 greedy tokens with each cache - the simulator should give exactly the same logits as the real backend.
    """)
    return


@app.cell
def s6_controls():
    RUN_S6 = True                 # set False to skip this section
    ADAPTER_WITH_SPARSE = False   # also run every flag with sink + local sparse attention
    SINK_TOKENS = 128             # sparse: always attend to the first SINK_TOKENS tokens
    LOCAL_WINDOW = 512            # sparse: plus the most recent LOCAL_WINDOW tokens
    ADAPTER_MAX_NEW_TOKENS = 32   # answer length for the adapter requests
    return (
        ADAPTER_MAX_NEW_TOKENS,
        ADAPTER_WITH_SPARSE,
        LOCAL_WINDOW,
        RUN_S6,
        SINK_TOKENS,
    )


@app.cell(hide_code=True)
def s6_context():
    KVQ_CONTEXT = (
        "The Amazon rainforest produces roughly 20 percent of the world's oxygen. "
        "It spans nine countries, with about 60 percent of it inside Brazil. "
        "The Amazon river discharges more water than any other river on Earth. "
    ) * 40
    return (KVQ_CONTEXT,)


@app.cell
def s6_adapter(
    ADAPTER_MAX_NEW_TOKENS,
    ADAPTER_WITH_SPARSE,
    HAS_MODELOPT,
    KVQ_CONTEXT,
    LOCAL_WINDOW,
    MODEL_NAME,
    ModelAdapterHF,
    RUN_S6,
    SINK_TOKENS,
    mo,
    model,
    time,
    torch,
):
    mo.stop(not RUN_S6, mo.md("_Skipped: RUN_S6 is False._"))
    from sparse_attention_hub.adapters import Request as _Request
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig as _ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        LocalMaskerConfig as _LocalMaskerConfig,
        SinkMaskerConfig as _SinkMaskerConfig,
    )

    _ = model  # needs the model cell
    _request = _Request(
        context=KVQ_CONTEXT,
        questions=["Which country contains most of the Amazon rainforest?", "What does the Amazon river discharge more of than any other river?"],
        answer_prefix="Answer: ",
    )
    _flags = [(False, {}), ("fake_nvfp4", {}), ("fake_nvfp4", {"axis_key": 0})]
    if HAS_MODELOPT:
        _flags += [("nvfp4", {}), ("nvfp4", {"axis_key": 0})]
    _sparse_options = [None] + ([_ResearchAttentionConfig([_SinkMaskerConfig(SINK_TOKENS), _LocalMaskerConfig(LOCAL_WINDOW)])] if ADAPTER_WITH_SPARSE else [])
    adapter_rows = []
    for _sparse in _sparse_options:
        for _flag, _kwargs in _flags:
            torch.cuda.reset_peak_memory_stats()
            _start = time.time()
            _adapter = ModelAdapterHF(
                MODEL_NAME, _sparse, model_kwargs={"dtype": torch.bfloat16}, device="cuda",
                quantize_kv_cache=_flag, kv_quantization_kwargs=_kwargs,
            )
            _answers = _adapter.process_request(_request, {"max_new_tokens": ADAPTER_MAX_NEW_TOKENS}, {"max_context_length": 8192}).responses
            adapter_rows.append({
                "quantize_kv_cache": repr(_flag), "layout": "KIVI" if _kwargs else "per token",
                "sparse": f"sink {SINK_TOKENS} + local {LOCAL_WINDOW}" if _sparse else "dense",
                "answer 1": _answers[0].strip()[:120], "answer 2": _answers[1].strip()[:120],
                "mentions Brazil": "brazil" in _answers[0].lower(),
                "seconds": round(time.time() - _start, 1),
                "peak GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            })
            del _adapter
            torch.cuda.empty_cache()
    check_adapter_answers = {
        "check": "adapter answers",
        "ok": all(r["answer 1"] and r["answer 2"] for r in adapter_rows),
        "detail": f"{len(adapter_rows)} configurations answered; Brazil in {sum(r['mentions Brazil'] for r in adapter_rows)}",
    }
    mo.vstack([mo.md("#### Through ModelAdapterHF"), mo.ui.table(adapter_rows, selection=None)])
    return (check_adapter_answers,)


@app.cell
def s6_greedy(
    CACHE_VARIANTS,
    KVQ_CONTEXT,
    RUN_S6,
    make_cache_factory,
    mo,
    model,
    tokenizer,
    torch,
):
    mo.stop(not RUN_S6)
    _prompt = tokenizer(KVQ_CONTEXT, return_tensors="pt").input_ids.cuda()
    _steps = 64

    def _greedy(make_cache):
        tokens, logits = [], []
        with torch.no_grad():
            out = model(_prompt, past_key_values=make_cache(), use_cache=True)
            for _ in range(_steps):
                step = out.logits[0, -1].float()
                logits.append(step)
                tokens.append(int(step.argmax()))
                out = model(torch.tensor([[tokens[-1]]], device="cuda"), past_key_values=out.past_key_values, use_cache=True)
        return tokens, torch.stack(logits)

    _runs = {}
    with mo.status.spinner(f"greedy decoding {_steps} tokens x {len(CACHE_VARIANTS)} caches ..."):
        for _label, _backend, _axis_key, _axis_value in CACHE_VARIANTS:
            _runs[_label] = _greedy(make_cache_factory(_backend, _axis_key, _axis_value))
    _bf16_tokens, _bf16_logits = _runs["bf16"]
    greedy_rows = []
    for _label, (_tokens, _logits) in _runs.items():
        _real = "KIVI nvfp4 (real)" if _label.startswith("KIVI") else "nvfp4 (real)"
        _row = {
            "cache": _label,
            "tokens = bf16": f"{sum(a == b for a, b in zip(_tokens, _bf16_tokens))}/{_steps}",
            "max |logit - bf16|": round((_logits - _bf16_logits).abs().max().item(), 3),
        }
        if _real in _runs:
            _row["tokens = real nvfp4"] = f"{sum(a == b for a, b in zip(_tokens, _runs[_real][0]))}/{_steps}"
            _row["max |logit - real|"] = (_logits - _runs[_real][1]).abs().max().item()
        greedy_rows.append(_row)
    _twins = [r for r in greedy_rows if "fake" in r["cache"] and "max |logit - real|" in r]
    check_greedy = {
        "check": "greedy: fake == real",
        "ok": all(r["max |logit - real|"] == 0.0 for r in _twins),
        "detail": f"{len(_twins)} layouts compared" if _twins else "real nvfp4 unavailable: nothing to compare",
    }
    mo.vstack([
        mo.md("#### Greedy decoding\n\nOnce one token differs the rest of the output diverges, so partial agreement just means not bit-identical, not worse."),
        mo.ui.table(greedy_rows, selection=None),
        mo.accordion({"bf16 continuation": mo.plain_text(tokenizer.decode(_bf16_tokens))}),
    ])
    return (check_greedy,)


@app.cell(hide_code=True)
def s6_summary(check_adapter_answers, check_greedy, kvq_summary):
    kvq_summary("6. Adapter + greedy", [check_adapter_answers, check_greedy])
    return


@app.cell(hide_code=True)
def s7_title(mo):
    mo.md(r"""
    ---
    ## 7. WikiText-2 perplexity

    2048-token windows from the WikiText-2 test set, one forward pass each with a fresh `quantize_prefill=True` cache. Earlier numbers on Qwen3-4B (20 windows): bf16 13.276, nvfp4 13.727 (fake and real), KIVI keys 13.440, keys + values per channel 13.263.
    """)
    return


@app.cell
def s7_controls():
    RUN_S7 = True     # set False to skip this section
    PPL_WINDOWS = 20  # 2048-token windows of WikiText-2 (about 145 = the whole test split)
    return PPL_WINDOWS, RUN_S7


@app.cell
def s7_perplexity(
    CACHE_VARIANTS,
    PPL_WINDOWS,
    RUN_S7,
    make_cache_factory,
    mo,
    model,
    perplexity,
    tokenizer,
    wikitext2_test_ids,
):
    mo.stop(not RUN_S7, mo.md("_Skipped: RUN_S7 is False._"))
    _ids = wikitext2_test_ids(tokenizer)
    _variants = CACHE_VARIANTS + [
        ("values per channel, fake_nvfp4", "fake_nvfp4", -1, 0),
        ("keys + values per channel, fake_nvfp4", "fake_nvfp4", 0, 0),
    ]
    ppl = {}
    with mo.status.progress_bar(total=len(_variants), title="perplexity") as _bar:
        for _label, _backend, _axis_key, _axis_value in _variants:
            _factory = make_cache_factory(_backend, _axis_key, _axis_value, quantize_prefill=True)
            ppl[_label] = perplexity(model, _ids, _factory, seq_len=2048, max_windows=PPL_WINDOWS)
            _bar.update(subtitle=f"{_label}: {ppl[_label]:.4f}")
    ppl_rows = [{"cache": k, "perplexity": round(v, 4), "vs bf16": f"{(v / ppl['bf16'] - 1) * 100:+.2f}%"} for k, v in ppl.items()]
    _gaps = [
        abs(ppl[fake] / ppl[real] - 1)
        for fake, real in (("fake_nvfp4", "nvfp4 (real)"), ("KIVI fake_nvfp4", "KIVI nvfp4 (real)"))
        if real in ppl
    ]
    check_ppl = {
        "check": "perplexity: fake == real, all finite",
        "ok": all(v == v and v < float("inf") for v in ppl.values()) and all(g < 0.005 for g in _gaps),
        "detail": f"{PPL_WINDOWS} windows; fake vs real gap {[f'{g:.2e}' for g in _gaps] if _gaps else 'n/a'}",
    }
    mo.ui.table(ppl_rows, selection=None)
    return check_ppl, ppl


@app.cell
def s7_plot(plt, ppl):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.barh(list(ppl), list(ppl.values()), color=["grey"] + ["tab:blue"] * (len(ppl) - 1))
    _ax.set_xlim(min(ppl.values()) * 0.98, max(ppl.values()) * 1.01)
    _ax.invert_yaxis()
    _ax.set_xlabel("WikiText-2 perplexity (lower is better)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s7_summary(check_ppl, kvq_summary):
    kvq_summary("7. Perplexity", [check_ppl])
    return


if __name__ == "__main__":
    app.run()
