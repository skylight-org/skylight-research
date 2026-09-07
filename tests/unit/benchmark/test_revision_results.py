"""Checkpoint revisions must not reuse another checkpoint's benchmark results."""

from pathlib import Path

import pytest

from benchmark.executor_config import (
    AdapterConfig,
    BenchmarkConfig,
    filter_existing_results,
    generate_benchmark_stubs,
)


def make_stubs(root, config, subsets):
    return generate_benchmark_stubs(
        model_names=["org/model"],
        sparse_attention_configs=[("dense", None)],
        benchmark_configs=[BenchmarkConfig("longbench", subsets)],
        adapter_config=config,
        base_result_dir=str(root),
    )


@pytest.mark.parametrize("subsets", [None, ["narrativeqa"]])
@pytest.mark.parametrize("field", ["revision", "model_kwargs", "tokenizer_kwargs"])
def test_different_revision_stays_pending(tmp_path, subsets, field):
    def config(revision):
        value = revision if field == "revision" else {"revision": revision}
        return AdapterConfig(**{field: value})

    first = make_stubs(tmp_path, config("stage/one"), subsets)
    result_dir = Path(first[0].result_dir)
    result_dir.mkdir(parents=True)
    (result_dir / "raw_results.csv").write_text("predicted_answer\nold\n")

    second = make_stubs(tmp_path, config("stage_one"), subsets)
    pending, completed = filter_existing_results(second, verbose=False)
    assert pending == second
    assert completed == []

    resumed = make_stubs(tmp_path, config("stage/one"), subsets)
    pending, completed = filter_existing_results(resumed, verbose=False)
    assert pending == []
    assert completed == resumed


def test_unpinned_result_path_is_unchanged(tmp_path):
    stub = make_stubs(tmp_path, AdapterConfig(), ["narrativeqa"])[0]
    assert Path(stub.result_dir) == tmp_path / "org_model/dense/longbench_narrativeqa"


def test_explicit_revisions_override_adapter_revision(tmp_path):
    configs = [
        AdapterConfig(
            revision=revision,
            model_kwargs={"revision": "weights"},
            tokenizer_kwargs={"revision": "tokenizer"},
        )
        for revision in ["ignored-a", "ignored-b"]
    ]
    assert (
        make_stubs(tmp_path, configs[0], None)[0].result_dir
        == make_stubs(tmp_path, configs[1], None)[0].result_dir
    )
