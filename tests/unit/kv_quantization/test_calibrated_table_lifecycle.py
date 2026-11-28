"""Calibrated table ownership, freeze lifetime, artifacts and failure atomicity."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    FullQuantizationCachePolicyConfig,
    KVQuantization,
    KVQuantizationConfig,
    NonUniformQuantizer,
    NonUniformQuantizerConfig,
    PrecisionPlan,
)
from sparse_attention_hub.kv_quantization.context import AttentionContext
from sparse_attention_hub.kv_quantization.quantizers.implementations import non_uniform

pytestmark = pytest.mark.unit
KEY = (0, "key")
VALUE = (1, "value")
CONTEXT = AttentionContext(0, "key", 0)


def vectors(offset=0):
    return torch.tensor([-2.0, -0.5, 0.5, 2.0])[:, None].repeat(1, 4) + offset


def fresh():
    return NonUniformQuantizer(
        NonUniformQuantizerConfig(
            mode="calibrated",
            calibration_iters=3,
        )
    )


def prepared(offset=0):
    quantizer = fresh()
    quantizer.calibrate({KEY: vectors(offset), VALUE: vectors(offset + 10)})
    return quantizer


def write_tables(tmp_path, *, tables=None, bits=2, name="tables.pt"):
    path = tmp_path / name
    torch.save(
        {"bits": bits, "tables": {"0:key": vectors().T} if tables is None else tables},
        path,
    )
    return str(path)


def assert_same_tables(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_first_encoding_preserves_expected_codes_and_freezes_the_codec():
    quantizer = prepared()
    assert not quantizer.tables_frozen
    x = torch.full((1, 1, 1, 4), 0.47)
    encoded = quantizer.quantize(x, context=CONTEXT)
    # Every coordinate is nearest to level 2 (=0.5): packed 10 10 10 10.
    assert encoded.codes.tolist() == [0b10101010]
    assert torch.equal(quantizer.dequantize(encoded), torch.full_like(x, 0.5))
    assert quantizer.tables_frozen
    assert len(encoded.metadata["table_fingerprint"]) == 64


@pytest.mark.parametrize("operation", ["calibrate", "load"])
def test_frozen_tables_reject_updates_without_changing_existing_entries(
    operation, tmp_path
):
    quantizer = prepared()
    x = vectors().view(1, 1, 4, 4)
    encoded = quantizer.quantize(x, context=CONTEXT)
    before = quantizer.dequantize(encoded).clone()
    tables = quantizer.tables
    path = write_tables(tmp_path, tables={"0:key": vectors(100).T})
    with pytest.raises(RuntimeError, match="frozen.*fresh quantizer"):
        if operation == "load":
            quantizer.load(path)
        else:
            quantizer.calibrate({KEY: vectors(100)})
    assert_same_tables(quantizer.tables, tables)
    assert torch.equal(quantizer.dequantize(encoded), before)
    assert quantizer.tables_frozen


def test_frozen_load_rejects_before_reading_another_file(tmp_path):
    quantizer = prepared()
    quantizer.quantize(vectors().view(1, 1, 4, 4), context=CONTEXT)
    with pytest.raises(RuntimeError, match="frozen"):
        quantizer.load(str(tmp_path / "does-not-exist.pt"))


def test_all_tables_can_be_prepared_and_replaced_before_first_encoding(tmp_path):
    quantizer = fresh()
    quantizer.calibrate({KEY: vectors()})
    quantizer.calibrate({VALUE: vectors(10)})
    quantizer.load(write_tables(tmp_path, tables={"0:key": vectors(100).T}))
    assert not quantizer.tables_frozen
    assert set(quantizer.tables) == {KEY, VALUE}
    x = torch.full((1, 1, 1, 4), 100.47)
    assert torch.equal(
        quantizer.fake_quantize(x, context=CONTEXT), torch.full_like(x, 100.5)
    )
    with pytest.raises(RuntimeError, match="frozen"):
        quantizer.calibrate({(2, "key"): vectors()})


@pytest.mark.parametrize("encode_first", [False, True])
def test_public_table_inspection_cannot_mutate_live_storage(encode_first):
    quantizer = prepared()
    x = torch.full((1, 1, 1, 4), 0.47)
    encoded = quantizer.quantize(x, context=CONTEXT) if encode_first else None
    view = quantizer.tables
    expected = view[KEY].clone()
    with pytest.raises(TypeError):
        view[KEY] = vectors(100).T
    with pytest.raises(AttributeError):
        quantizer.tables = {}
    view[KEY].fill_(9999)  # Inspection tensors are detached copies.
    assert torch.equal(quantizer.tables[KEY], expected)
    if encoded is None:
        encoded = quantizer.quantize(x, context=CONTEXT)
    assert torch.equal(quantizer.dequantize(encoded), torch.full_like(x, 0.5))


def test_load_owns_copies_even_when_deserialized_tensors_are_shared(monkeypatch):
    source = vectors().T.contiguous()
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"bits": 2, "tables": {"0:key": source}}
    )
    quantizer = fresh()
    quantizer.load("unused.pt")
    source.fill_(9999)
    x = torch.full((1, 1, 1, 4), 0.47)
    assert torch.equal(
        quantizer.fake_quantize(x, context=CONTEXT), torch.full_like(x, 0.5)
    )


@pytest.mark.parametrize(
    "failure", ["context", "key", "channels", "input", "plan", "packing"]
)
def test_failed_first_encoding_does_not_freeze_tables(failure, monkeypatch):
    quantizer = prepared()
    x = torch.ones(1, 1, 1, 4)
    context = CONTEXT
    plan = None
    if failure == "context":
        context = None
    elif failure == "key":
        context = AttentionContext(9, "key", 0)
    elif failure == "channels":
        x = torch.ones(1, 1, 1, 8)
    elif failure == "input":
        x.fill_(float("nan"))
    elif failure == "plan":
        plan = PrecisionPlan(bits=4)
    with monkeypatch.context() as scoped:
        if failure == "packing":

            def fail(*args, **kwargs):
                raise RuntimeError("simulated packing failure")

            scoped.setattr(non_uniform, "pack_codes", fail)
        with pytest.raises((ValueError, RuntimeError)):
            quantizer.quantize(x, plan, context=context)
    assert not quantizer.tables_frozen
    quantizer.calibrate({KEY: vectors(100)})
    encoded = quantizer.quantize(torch.full((1, 1, 1, 4), 100.47), context=CONTEXT)
    assert torch.equal(quantizer.dequantize(encoded), torch.full((1, 1, 1, 4), 100.5))


def test_save_after_freezing_round_trips_and_matching_decode_freezes_receiver(tmp_path):
    quantizer = prepared()
    x = vectors().view(1, 1, 4, 4)
    encoded = quantizer.quantize(x, context=CONTEXT)
    path = str(tmp_path / "frozen.pt")
    quantizer.save(path)
    receiver = NonUniformQuantizer(
        NonUniformQuantizerConfig(mode="calibrated", codebook_path=path)
    )
    assert not receiver.tables_frozen
    assert torch.equal(receiver.dequantize(encoded), quantizer.dequantize(encoded))
    assert receiver.tables_frozen
    with pytest.raises(RuntimeError, match="frozen"):
        receiver.calibrate({KEY: vectors(100)})
    assert quantizer.tables_frozen


def test_wrong_table_cannot_silently_decode_another_codecs_entries():
    first, second = prepared(), prepared(100)
    encoded = first.quantize(vectors().view(1, 1, 4, 4), context=CONTEXT)
    with pytest.raises(ValueError, match="does not match"):
        second.dequantize(encoded)
    assert not second.tables_frozen
    second.calibrate({KEY: vectors(200)})


def test_record_without_table_identity_is_rejected():
    quantizer = prepared()
    encoded = quantizer.quantize(vectors().view(1, 1, 4, 4), context=CONTEXT)
    encoded.metadata.pop("table_fingerprint")
    with pytest.raises(ValueError, match="does not match"):
        quantizer.dequantize(encoded)


def test_request_cleanup_does_not_thaw_a_used_quantizer():
    quant = KVQuantization(
        KVQuantizationConfig(
            key_quantizer=NonUniformQuantizerConfig(
                mode="calibrated", calibration_iters=3
            ),
            cache_policy=FullQuantizationCachePolicyConfig(),
            store_codes=True,
        )
    )
    quant.key_quantizer.calibrate({KEY: vectors()})
    metadata = {}
    x = vectors().view(1, 1, 4, 4)
    first, _ = quant.process(0, x, x, metadata, num_new_tokens=0)
    assert CACHE_KEY in metadata
    quant.clear_request(metadata)
    assert CACHE_KEY not in metadata
    with pytest.raises(RuntimeError, match="frozen"):
        quant.key_quantizer.calibrate({KEY: vectors(100)})
    again, _ = quant.process(0, x, x, {}, num_new_tokens=0)
    assert torch.equal(first, again)


INVALID_TABLES = [
    pytest.param(None, id="not-tensor"),
    pytest.param(torch.ones(4, 4, dtype=torch.int64), id="integer"),
    pytest.param(torch.ones(4), id="rank"),
    pytest.param(torch.ones(0, 4), id="empty-channels"),
    pytest.param(torch.ones(4, 3), id="level-count"),
    pytest.param(torch.full((4, 4), float("nan")), id="nan"),
    pytest.param(torch.full((4, 4), float("inf")), id="infinity"),
    pytest.param(torch.full((4, 4), 1e300, dtype=torch.float64), id="fp32-overflow"),
    pytest.param(vectors().T.flip(-1), id="unsorted"),
    pytest.param(vectors().T.to_sparse(), id="sparse"),
    pytest.param(torch.ones(4, 4, dtype=torch.complex64), id="complex"),
]


@pytest.mark.parametrize("invalid", INVALID_TABLES)
def test_invalid_table_file_is_rejected_atomically(invalid, tmp_path):
    quantizer = prepared()
    before = quantizer.tables
    path = write_tables(tmp_path, tables={"0:key": vectors(100).T, "1:value": invalid})
    with pytest.raises(ValueError):
        quantizer.load(path)
    assert_same_tables(quantizer.tables, before)
    assert not quantizer.tables_frozen


@pytest.mark.parametrize(
    "name", ["bad", "-1:key", "0:query", "00:key", "0:key:extra", (0, "key"), 0]
)
def test_invalid_table_names_are_rejected_atomically(name, tmp_path):
    quantizer = prepared()
    before = quantizer.tables
    path = write_tables(tmp_path, tables={"0:key": vectors(100).T, name: vectors().T})
    with pytest.raises(ValueError):
        quantizer.load(path)
    assert_same_tables(quantizer.tables, before)


@pytest.mark.parametrize(
    "artifact",
    [
        None,
        {},
        {"bits": True, "tables": {}},
        {"bits": "2", "tables": {}},
        {"bits": 2},
        {"bits": 2, "tables": []},
        {"bits": 2, "tables": {}},
    ],
)
def test_invalid_file_structure_does_not_change_prepared_tables(artifact, tmp_path):
    quantizer = prepared()
    before = quantizer.tables
    path = tmp_path / "invalid.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError):
        quantizer.load(str(path))
    assert_same_tables(quantizer.tables, before)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_loaded_levels_are_owned_fp32_and_duplicate_levels_are_allowed(dtype, tmp_path):
    quantizer = fresh()
    path = write_tables(tmp_path, tables={"0:key": torch.ones(4, 4, dtype=dtype)})
    quantizer.load(path)
    assert quantizer.tables[KEY].dtype == torch.float32
    x = torch.ones(1, 1, 1, 4)
    assert torch.equal(quantizer.fake_quantize(x, context=CONTEXT), x)


@pytest.mark.parametrize(
    "bad_samples",
    [
        {},
        [],
        {KEY: torch.empty(0, 4)},
        {KEY: torch.tensor(1.0)},
        {KEY: torch.ones(4, 4, dtype=torch.int64)},
        {KEY: None},
        {KEY: torch.full((4, 4), float("nan"))},
        {KEY: torch.full((4, 4), 1e300, dtype=torch.float64)},
        {(-1, "key"): vectors()},
        {(True, "key"): vectors()},
        {(0, "query"): vectors()},
        {"0:key": vectors()},
    ],
)
def test_invalid_calibration_samples_preserve_existing_tables(bad_samples):
    quantizer = prepared()
    before = quantizer.tables
    with pytest.raises(ValueError):
        quantizer.calibrate(bad_samples)
    assert_same_tables(quantizer.tables, before)
    assert not quantizer.tables_frozen


def test_calibration_validation_is_atomic_across_multiple_layers():
    quantizer = prepared()
    before = quantizer.tables
    with pytest.raises(ValueError):
        quantizer.calibrate({KEY: vectors(100), VALUE: torch.empty(0, 4)})
    assert_same_tables(quantizer.tables, before)


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5])
def test_invalid_calibration_iteration_counts_are_rejected(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        NonUniformQuantizer(
            NonUniformQuantizerConfig(mode="calibrated", calibration_iters=iterations)
        )


def test_gaussian_encoding_stays_unfrozen_and_does_not_accept_calibrated_files(
    tmp_path,
):
    quantizer = NonUniformQuantizer(NonUniformQuantizerConfig())
    quantizer.quantize(torch.ones(1, 1, 1, 4))
    assert not quantizer.tables_frozen
    with pytest.raises(ValueError, match="calibrated"):
        quantizer.load(write_tables(tmp_path))


def test_saving_without_prepared_tables_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="prepared calibrated"):
        fresh().save(str(tmp_path / "empty.pt"))


@pytest.mark.parametrize("frozen", [False, True])
def test_worker_serialization_preserves_tables_and_freeze_state(frozen):
    quantizer = prepared()
    x = vectors().view(1, 1, 4, 4)
    encoded = quantizer.quantize(x, context=CONTEXT) if frozen else None
    worker_copy = pickle.loads(pickle.dumps(quantizer))
    assert worker_copy.tables_frozen == frozen
    assert_same_tables(worker_copy.tables, quantizer.tables)
    if encoded is None:
        encoded = worker_copy.quantize(x, context=CONTEXT)
        quantizer.calibrate({KEY: vectors(100)})
    else:
        with pytest.raises(RuntimeError, match="frozen"):
            worker_copy.calibrate({KEY: vectors(100)})
    assert torch.equal(worker_copy.dequantize(encoded), x)


@pytest.mark.parametrize("operation", ["calibrate", "load"])
def test_table_update_cannot_race_with_first_successful_encoding(
    operation, tmp_path, monkeypatch
):
    quantizer = prepared()
    x = vectors().view(1, 1, 4, 4)
    encoding_started, release_encoding, update_attempted = Event(), Event(), Event()
    actual_lock = quantizer._tables_lock

    class ObservedLock:
        def __enter__(self):
            if current_thread().name.startswith("table-update"):
                update_attempted.set()
            actual_lock.acquire()

        def __exit__(self, *args):
            actual_lock.release()

    monkeypatch.setattr(quantizer, "_tables_lock", ObservedLock())
    actual_pack = non_uniform.pack_codes

    def paused_pack(*args, **kwargs):
        encoding_started.set()
        if not release_encoding.wait(5):
            raise RuntimeError("test did not release encoding")
        return actual_pack(*args, **kwargs)

    monkeypatch.setattr(non_uniform, "pack_codes", paused_pack)
    path = write_tables(tmp_path, tables={"0:key": vectors(100).T})

    def update():
        if operation == "load":
            quantizer.load(path)
        else:
            quantizer.calibrate({KEY: vectors(100)})

    with ThreadPoolExecutor(1, thread_name_prefix="table-encode") as encoder:
        with ThreadPoolExecutor(1, thread_name_prefix="table-update") as updater:
            encoding = encoder.submit(quantizer.quantize, x, context=CONTEXT)
            try:
                assert encoding_started.wait(5)
                change = updater.submit(update)
                assert update_attempted.wait(5)
                assert not change.done()  # Blocked until first encoding finishes.
            finally:
                release_encoding.set()
            encoded = encoding.result(timeout=5)
            with pytest.raises(RuntimeError, match="frozen"):
                change.result(timeout=5)
    assert torch.equal(quantizer.dequantize(encoded), x)
