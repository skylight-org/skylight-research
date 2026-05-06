# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared defaults for AIME2024 / AIME2025 benchmarks."""

# Upper bound merged with ``generation_kwargs`` in ``Benchmark._process_all_requests`` via
# ``min(param_max_new_tokens, row_max_new_tokens)``. Must be >= any intended CLI value.
MAX_NEW_TOKENS_CAP: int = 65536
