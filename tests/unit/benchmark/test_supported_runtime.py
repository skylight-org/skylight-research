"""Installed package metadata and the HF attention integration contract."""

import sys
from importlib.metadata import metadata, version

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

pytestmark = pytest.mark.unit


def test_package_metadata_rejects_python_without_keyword_only_dataclass_fields():
    package = metadata("sparse-attention-hub")
    supported = SpecifierSet(package["Requires-Python"])
    assert "3.9" not in supported
    assert "3.10" in supported
    assert (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        in supported
    )


def test_dependency_metadata_matches_the_imported_hf_registry_contract():
    package = metadata("sparse-attention-hub")
    requirements = [Requirement(item) for item in package.get_all("Requires-Dist", [])]
    requirement = next(item for item in requirements if item.name == "transformers")
    assert version("transformers") in requirement.specifier
    assert "4.20.0" not in requirement.specifier
    # Import the exact public/internal integration entry points the adapter uses.
    from transformers import DynamicCache
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert callable(DynamicCache)
    assert callable(ALL_ATTENTION_FUNCTIONS.register)
    assert callable(ALL_MASK_ATTENTION_FUNCTIONS.register)
    assert ALL_MASK_ATTENTION_FUNCTIONS["eager"] is eager_mask
