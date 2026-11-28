"""Data processing utilities for benchmark execution.

This module contains utility functions for data processing, formatting, and export.
"""

import csv
import dataclasses
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def make_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to strings recursively, including dataclasses.

    Args:
        obj: Object to make JSON serializable

    Returns:
        JSON serializable version of the object

    Example:
        >>> config = {"torch_dtype": torch.bfloat16, "device": "cuda"}
        >>> serializable = make_serializable(config)
        >>> json.dumps(serializable)  # Works without error
    """
    if isinstance(obj, Enum):
        return make_serializable(obj.value)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # asdict() recursively erases nested config classes. Preserve their
        # identities so a KIVI preset can be distinguished from a generic config.
        return {
            "__type__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            **{
                field.name: make_serializable(getattr(obj, field.name))
                for field in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    elif hasattr(obj, "dtype"):
        return str(obj)
    elif isinstance(obj, dict):
        return {
            (
                str(make_serializable(k)) if isinstance(k, Enum) else k
            ): make_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (set, frozenset)):
        return sorted(
            [make_serializable(item) for item in obj],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    elif hasattr(obj, "__dict__"):
        # For custom objects
        return {
            "__type__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            **make_serializable(vars(obj)),
        }
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            # Default object reprs contain memory addresses, which change between
            # runs. An opaque object's stable class name is the available identity.
            return {"__type__": f"{type(obj).__module__}.{type(obj).__qualname__}"}


def escape_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Escape special characters in DataFrame for safe CSV export.

    This function prepares a DataFrame for CSV export by escaping newlines,
    carriage returns, and other special characters that could cause parsing issues.

    Args:
        df: DataFrame to escape

    Returns:
        DataFrame with escaped string columns

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['Line 1\nLine 2', 'Text with, comma']
        ... })
        >>> escaped_df = escape_dataframe_for_csv(df)
        >>> escaped_df.to_csv('output.csv', quoting=csv.QUOTE_ALL)
    """
    escaped_df = df.copy()

    for col in escaped_df.columns:
        if escaped_df[col].dtype == "object":  # Only process string columns
            escaped_df[col] = (
                escaped_df[col]
                .astype(str)
                .str.replace("\n", "\\n")
                .str.replace("\r", "\\r")
            )

    return escaped_df


def save_dataframe_to_csv(
    df: pd.DataFrame, file_path: str, index: bool = False, quoting: int = csv.QUOTE_ALL
) -> None:
    """Save DataFrame to CSV with proper escaping and quoting.

    This is a convenience function that combines escaping and CSV export
    with consistent settings for benchmark results.

    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
        index: Whether to include DataFrame index in output
        quoting: CSV quoting mode (default: QUOTE_ALL)

    Example:
        >>> df = pd.DataFrame({'text': ['Line 1\nLine 2']})
        >>> save_dataframe_to_csv(df, 'results.csv')
    """
    escaped_df = escape_dataframe_for_csv(df)
    escaped_df.to_csv(file_path, index=index, quoting=quoting)
