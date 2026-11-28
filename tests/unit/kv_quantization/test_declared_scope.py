"""Unavailable steps fail when selected instead of producing misleading scores."""

import pytest

from sparse_attention_hub.kv_quantization import KVQuantization, KVQuantizationConfig

pytestmark = pytest.mark.unit


def test_coordinator_calibration_is_not_an_implemented_recipe_pipeline():
    with pytest.raises(NotImplementedError, match="Calibration recipes"):
        KVQuantization.create_from_config(KVQuantizationConfig()).calibrate({})
