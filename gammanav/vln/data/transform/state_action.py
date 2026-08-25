from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr, field_validator
import torch

from gammanav.vln.data.schema import DatasetMetadata
from gammanav.vln.data.transform.base import InvertibleModalityTransform


class Q99Normalizer:
    """Map the 1st/99th percentile interval to [-1, 1]."""

    def __init__(self, statistics: dict):
        self.q01 = torch.as_tensor(statistics["q01"])
        self.q99 = torch.as_tensor(statistics["q99"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(dtype=value.dtype, device=value.device)
        q99 = self.q99.to(dtype=value.dtype, device=value.device)
        valid = q01 != q99
        normalized = torch.zeros_like(value)
        normalized[..., valid] = (
            2
            * (value[..., valid] - q01[..., valid])
            / (q99[..., valid] - q01[..., valid])
            - 1
        )
        normalized[..., ~valid] = value[..., ~valid]
        return normalized.clamp(-1, 1)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(dtype=value.dtype, device=value.device)
        q99 = self.q99.to(dtype=value.dtype, device=value.device)
        return (value + 1) / 2 * (q99 - q01) + q01


class StateActionToTensor(InvertibleModalityTransform):
    """Convert navigation state/action arrays to tensors and back."""

    input_dtypes: dict[str, np.dtype] = Field(default_factory=dict)
    output_dtypes: dict[str, torch.dtype] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        include = (
            {"apply_to"}
            if kwargs.get("mode", "python") == "json"
            else kwargs.pop("include", None)
        )
        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, values):
        for key, dtype in values.items():
            if not isinstance(dtype, str):
                continue
            dtype_name = dtype.split(".")[-1]
            if dtype.startswith("torch."):
                values[key] = getattr(torch, dtype_name)
            elif dtype.startswith(("np.", "numpy.")):
                values[key] = np.dtype(dtype_name)
            else:
                raise ValueError(f"Invalid dtype: {dtype}")
        return values

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{key} must be a numpy array, got {type(value)}")
            if not value.flags.writeable:
                value = value.copy()
            tensor = torch.from_numpy(value)
            data[key] = (
                tensor.to(self.output_dtypes[key])
                if key in self.output_dtypes
                else tensor
            )
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a tensor, got {type(value)}")
            array = value.numpy()
            data[key] = (
                array.astype(self.input_dtypes[key])
                if key in self.input_dtypes
                else array
            )
        return data


class StateActionTransform(InvertibleModalityTransform):
    """Apply WNM-3D's q99 navigation-state/action normalization."""

    normalization_modes: dict[str, str] = Field(default_factory=dict)
    _normalizers: dict[str, Q99Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, torch.dtype] = PrivateAttr(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        include = (
            {"apply_to", "normalization_modes"}
            if kwargs.get("mode", "python") == "json"
            else kwargs.pop("include", None)
        )
        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("normalization_modes")
    def validate_modes(cls, modes):
        invalid = {key: mode for key, mode in modes.items() if mode != "q99"}
        if invalid:
            raise ValueError(f"WNM-3D only supports q99 normalization, got {invalid}")
        return modes

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        for key in self.normalization_modes:
            modality, subkey = key.split(".", 1)
            statistics = getattr(dataset_metadata.statistics, modality)
            if subkey not in statistics:
                raise KeyError(f"Statistics for {key} are missing")
            self._normalizers[key] = Q99Normalizer(statistics[subkey].model_dump())

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a tensor, got {type(value)}")
            self._input_dtypes.setdefault(key, value.dtype)
            if value.dtype != self._input_dtypes[key]:
                raise TypeError(f"Inconsistent dtype for {key}: {value.dtype}")
            data[key] = (
                self._normalizers[key].forward(value)
                if key in self._normalizers
                else value
            )
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a tensor, got {type(value)}")
            if key in self._normalizers:
                value = self._normalizers[key].inverse(value)
            if key in self._input_dtypes:
                value = value.to(self._input_dtypes[key])
            data[key] = value
        return data
