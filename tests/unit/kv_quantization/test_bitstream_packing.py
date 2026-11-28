"""Cross-byte storage checked against independent Python integer arithmetic."""

import pytest
import torch

from sparse_attention_hub.kv_quantization.quantizers.common import (
    pack_bitstream,
    pack_codes,
    unpack_bitstream,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 8, 9, 31, 129])
def test_bitstream_matches_independent_bytes_and_codes(bits, count):
    values = torch.randint(
        0, 2**bits, (count,), generator=torch.Generator().manual_seed(341)
    )
    packed, actual_count = pack_bitstream(values, bits)
    integer = sum(int(value) << (index * bits) for index, value in enumerate(values))
    expected = integer.to_bytes((count * bits + 7) // 8, "little")
    assert packed.tolist() == list(expected)
    assert actual_count == count
    assert torch.equal(unpack_bitstream(packed, bits, count).long(), values)
    if bits in (1, 2, 4, 8):
        assert torch.equal(packed, pack_codes(values, bits)[0])


def test_eight_three_bit_codes_take_three_bytes():
    packed, count = pack_bitstream(torch.arange(8), 3)
    assert packed.numel() == 3
    assert unpack_bitstream(packed, 3, count).tolist() == list(range(8))


def test_full_uint8_range_does_not_wrap_the_code_bound_comparison():
    values = torch.arange(256, dtype=torch.uint8)
    packed, count = pack_bitstream(values, 8)
    assert torch.equal(packed, values)
    assert torch.equal(unpack_bitstream(packed, 8, count).to(torch.uint8), values)


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([-1]),
        torch.tensor([8]),
        torch.tensor([0.5]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([1 + 2j]),
    ],
)
def test_invalid_codes_are_rejected(values):
    with pytest.raises(ValueError):
        pack_bitstream(values, 3)


@pytest.mark.parametrize("bits", [0, 9, True, 3.0])
def test_invalid_bit_widths_are_rejected(bits):
    with pytest.raises(ValueError):
        pack_bitstream(torch.tensor([0]), bits)
    with pytest.raises(ValueError):
        unpack_bitstream(torch.tensor([0], dtype=torch.uint8), bits, 1)


@pytest.mark.parametrize(
    "packed,count",
    [
        (torch.tensor([128], dtype=torch.uint8), 1),
        (torch.tensor([], dtype=torch.uint8), 1),
        (torch.tensor([0], dtype=torch.int16), 1),
        (torch.tensor([[0]], dtype=torch.uint8), 1),
        (torch.tensor([0], dtype=torch.uint8), -1),
        (torch.tensor([0], dtype=torch.uint8), True),
    ],
)
def test_invalid_stream_count_dtype_and_padding_are_rejected(packed, count):
    with pytest.raises(ValueError):
        unpack_bitstream(packed, 3, count)
