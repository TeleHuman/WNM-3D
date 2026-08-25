# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .embodiment_tags import EmbodimentTag

# Common schema


# LeRobot schema


class LeRobotModalityField(BaseModel):
    """Metadata for a LeRobot modality field."""

    original_key: Optional[str] = Field(
        default=None,
        description="The original key of the modality in the LeRobot dataset",
    )


class LeRobotModalityMetadata(BaseModel):
    """Video and language metadata used by the InteriorGS loader."""

    video: dict[str, LeRobotModalityField] = Field(
        ...,
        description="The metadata for the video modality. The keys are the new names of each video modality.",
    )
    annotation: Optional[dict[str, LeRobotModalityField]] = Field(
        default=None,
        description="The metadata for the annotation modality. The keys are the new names of each annotation modality.",
    )

    @model_validator(mode="after")
    def check_original_keys(self):
        for key in self.video:
            if self.video[key].original_key is None:
                self.video[key].original_key = "observation.images." + key

        if self.annotation is not None:
            for key in self.annotation:
                if self.annotation[key].original_key is None:
                    self.annotation[key].original_key = "annotation." + key

        return self

    def get_key_meta(self, key: str) -> LeRobotModalityField:
        """Return metadata for an InteriorGS video or language key."""
        split_key = key.split(".")
        modality = split_key[0]
        subkey = ".".join(split_key[1:])
        if modality == "video":
            if subkey not in self.video:
                raise ValueError(
                    f"Key: {key}, video key {subkey} not found in metadata, available video keys: {self.video.keys()}"
                )
            return self.video[subkey]
        elif modality == "annotation":
            assert self.annotation is not None, (
                "Trying to get annotation metadata for a dataset with no annotations"
            )
            if subkey not in self.annotation:
                raise ValueError(
                    f"Key: {key}, annotation key {subkey} not found in metadata, available annotation keys: {self.annotation.keys()}"
                )
            return self.annotation[subkey]
        else:
            raise ValueError(f"Key: {key}, unexpected modality: {modality}")


# Dataset schema (parsed from LeRobot schema and simplified)


class DatasetStatisticalValues(BaseModel):
    max: np.ndarray = Field(..., description="Maximum values")
    min: np.ndarray = Field(..., description="Minimum values")
    mean: np.ndarray = Field(..., description="Mean values")
    std: np.ndarray = Field(..., description="Standard deviation")
    q01: np.ndarray = Field(..., description="1st percentile values")
    q99: np.ndarray = Field(..., description="99th percentile values")

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: np.ndarray) -> list:
        return v.tolist()  # type: ignore

    @field_validator("*", mode="before")
    def validate_ndarray(cls, v) -> np.ndarray:
        return np.array(v)


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(
        ..., description="Statistics of the state"
    )
    action: dict[str, DatasetStatisticalValues] = Field(
        ..., description="Statistics of the action"
    )


class VideoMetadata(BaseModel):
    """Metadata of the video modality"""

    resolution: tuple[int, int] = Field(..., description="Resolution of the video")
    channels: int = Field(..., description="Number of channels in the video", gt=0)
    fps: float = Field(..., description="Frames per second", gt=0)


class StateActionMetadata(BaseModel):
    shape: tuple[int, ...] = Field(..., description="Shape of the state or action")


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata] = Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = Field(
        ..., description="Metadata of the state"
    )
    action: dict[str, StateActionMetadata] = Field(
        ..., description="Metadata of the action"
    )


class DatasetMetadata(BaseModel):
    """Statistics and modality shapes used by WNM-3D transforms."""

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(
        ..., description="Embodiment tag of the dataset"
    )
