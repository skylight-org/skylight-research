"""Unavailable recipes fail at selection instead of producing misleading scores."""

import pytest

from sparse_attention_hub.kv_quantization import (
    KVQuantization,
    KVQuantizationConfig,
    KVQuantizer,
    KVQuantNUQConfig,
    KVTransform,
    LearnedRotationConfig,
    NVFP4KVConfig,
    NVFP4QuantizerConfig,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "config",
    [
        KVQuantNUQConfig(),
        NVFP4KVConfig(),
    ],
)
def test_pending_recipe_selection_fails_before_model_execution(config):
    with pytest.raises(NotImplementedError, match="recipe scaffold"):
        KVQuantization.create_from_config(config)


def test_native_nvfp4_and_learned_rotation_are_explicitly_unavailable():
    with pytest.raises(NotImplementedError, match="scaffold"):
        KVQuantizer.create_from_config(NVFP4QuantizerConfig())
    with pytest.raises(NotImplementedError, match="scaffold"):
        KVTransform.create_from_config(LearnedRotationConfig())


def test_coordinator_calibration_is_not_an_implemented_recipe_pipeline():
    with pytest.raises(NotImplementedError, match="Calibration recipes"):
        KVQuantization.create_from_config(KVQuantizationConfig()).calibrate({})
