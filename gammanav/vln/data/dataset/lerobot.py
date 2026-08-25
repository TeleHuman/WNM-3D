from collections import defaultdict
import copy
import glob
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Sequence, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml

from gammanav.vln.data.schema import (
    DatasetMetadata,
    EmbodimentTag,
    LeRobotModalityMetadata,
)
from gammanav.vln.data.transform import ComposedModalityTransform

T_LeRobotMixtureDataset = TypeVar(
    "T_LeRobotMixtureDataset", bound="LeRobotMixtureDataset"
)

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"


class ModalityConfig(BaseModel):
    """Relative sample indices and keys for one WNM-3D input modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    eval_delta_indices: list[int] | None = None
    """Delta indices to sample relative to the current index for evaluation. If None, uses the same indices as delta_indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

    def model_post_init(self, *args, **kwargs):
        """Initialize eval_delta_indices to delta_indices if not provided."""
        super().model_post_init(*args, **kwargs)
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


class LeRobotSingleDataset(Dataset):
    """Shared LeRobot indexing and video/language loading for InteriorGS."""

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        transforms: ComposedModalityTransform | None = None,
        discard_bad_trajectories: bool = True,
        max_chunk_size: int = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.max_chunk_size = max_chunk_size
        self.transforms = (
            transforms
            if transforms is not None
            else ComposedModalityTransform(transforms=[])
        )
        self.discard_bad_trajectories = discard_bad_trajectories
        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        self.tag = EmbodimentTag(embodiment_tag)
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()

        # Initialize trajectory info and chunk size before building metadata.
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._data_path_pattern = self._get_data_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._metadata = self._get_metadata()
        self._step_filter = self._get_step_filter()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()
        self._dataset_name = self._dataset_path.name

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        self._video_path_pattern = self._get_video_path_pattern()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """Keys grouped by configured WNM-3D modality."""
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.

        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        """The step filter for the dataset."""
        return self._step_filter

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Get the step filter for the dataset."""
        step_filter_path = self.dataset_path / STEP_FILTER_FILENAME
        step_filter = {}
        if step_filter_path.exists():
            with open(step_filter_path, "r") as f:
                for line in f:
                    episode_step_filter = json.loads(line)
                    trajectory_id = episode_step_filter["episode_index"]
                    all_indices = np.arange(
                        self.trajectory_lengths[trajectory_id].item()
                    )
                    indices_to_filter = np.array(episode_step_filter["step_indices"])
                    step_filter[trajectory_id] = np.setdiff1d(
                        all_indices, indices_to_filter
                    )
        else:
            for trajectory_id in self.trajectory_ids:
                step_filter[trajectory_id] = np.arange(
                    self.trajectory_lengths[trajectory_id].item()
                )
        return step_filter

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[int, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.step_filter: {
                0: [0, 1, 2],
                1: [0, 1],
                2: [0, 2, 3]
            }
            return: [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1),
                (2, 0), (2, 2), (2, 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        # All steps is used in single dataset, so we need to discard bad trajectories
        # Mixture dataset directly use trajectory_ids, so we handle it by changing the sampling weights
        discarded_episode_indices = []
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get(
                "discarded_episode_indices", []
            )

        for trajectory_id in self.trajectory_ids:
            if trajectory_id in discarded_episode_indices:
                continue
            for base_index in self.step_filter[trajectory_id]:
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.

        Returns:
            dict: Dictionary mapping modality names to their keys.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        indices = {
            key: delta_indices + base_index
            for key, delta_indices in self.delta_indices.items()
        }
        return self.transforms(self.get_step_data(trajectory_id, indices))

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """Load untransformed WNM-3D modalities for one trajectory step."""
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
        return data

    def get_parquet_path(self, trajectory_id: int) -> Path:
        """Get the parquet path for a trajectory."""
        chunk_index = self.get_episode_chunk(trajectory_id)
        return self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            parquet_path = self.get_parquet_path(trajectory_id)
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """Get the video file path for a specific trajectory and video key.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The video key (without 'video.' prefix).

        Returns:
            Path: Path to the video file.
        """
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index,
            episode_index=trajectory_id,
            video_key=original_key,
        )
        return self.dataset_path / video_filename

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            list[str]: The annotation data for the trajectory and step indices.
                If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        assert key.startswith("annotation."), (
            f"Language key must start with 'annotation.', got {key}"
        )
        subkey = key.replace("annotation.", "")

        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert subkey in annotation_meta, (
            f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        )
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        if pd.api.types.is_numeric_dtype(self.curr_traj_data[original_key]):
            # Stored as list of integers
            task_indices: list[int] = (
                self.curr_traj_data[original_key].iloc[step_indices].tolist()
            )
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            # Stored as list of strings
            return (
                self.curr_traj_data[original_key]
                .iloc[step_indices]
                .astype(str)
                .tolist()
            )


def safe_hash(input_tuple):
    """Generate a safe hash from an input tuple.

    Creates a deterministic hash using SHA256 and returns the lower 128 bits.
    This is used for deterministic random seed generation.

    Args:
        input_tuple: The tuple to hash.

    Returns:
        int: A 128-bit hash value.
    """
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    """Specification element for a dataset mixture defining paths and weights.

    This class validates dataset paths by embodiment tag and handles weight distribution
    across multiple dataset paths if requested.
    """

    dataset_path: dict[str, list[Path] | Path] = Field(
        ..., description="The path to the dataset."
    )
    dataset_weight: float = Field(
        ..., description="The weight of the dataset in the mixture."
    )
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )

    @field_validator("dataset_path", mode="after")
    def validate_dataset_path_keys(
        cls, v: dict[str, list[Path] | Path]
    ) -> dict[str, list[Path]]:
        """Validate dataset paths and expand glob patterns.

        Args:
            v (dict[str, list[Path] | Path]): Dictionary mapping embodiment tags to paths.

        Returns:
            dict[str, list[Path]]: Validated and expanded paths.

        Raises:
            ValueError: If an invalid embodiment tag is provided.
        """
        all_globbed_paths: dict[str, list[Path]] = {}
        for embodiment_tag, paths in v.items():
            try:
                _ = EmbodimentTag(embodiment_tag)
            except ValueError:
                raise ValueError(f"Invalid embodiment tag: {embodiment_tag}")
            if isinstance(paths, Path):
                paths = [paths]
            globbed_paths = []
            for path in paths:
                globbed_paths.extend(glob.glob(str(path)))
            all_globbed_paths[embodiment_tag] = globbed_paths
        return all_globbed_paths


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        allow_padding_at_end: bool = False,
        metadata_config: dict | None = None,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            training (bool): If True, __getitem__ will return different samples every epoch; if False, __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            allow_padding_at_end (bool): If True, allow padding at the end of the dataset.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.training = training
        self.allow_padding_at_end = allow_padding_at_end

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_ids))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= np.array(
                    [
                        len(dataset.step_filter[trajectory_id])
                        for trajectory_id in dataset.trajectory_ids
                    ]
                )

            if dataset.discard_bad_trajectories:
                bad_trajectory_indices = dataset.lerobot_info_meta.get(
                    "discarded_episode_indices", []
                )
                trajectory_sampling_weights[bad_trajectory_indices] = 0.0

            if trajectory_sampling_weights.sum() == 0:
                raise ValueError(f"No valid trajectories found for dataset {dataset}")

            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        if metadata_config is None:
            metadata_config = {"percentile_mixing_method": "min_max"}
        self.update_metadata(metadata_config)

        # Set the transforms to training or evaluation mode
        if self.training:
            for dataset in self.datasets:
                dataset.transforms.train()
        else:
            for dataset in self.datasets:
                dataset.transforms.eval()

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    def __str__(self) -> str:
        """Return a string representation of the mixture dataset with weights."""
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump({"Mixture dataset": dataset_descriptions})

    @classmethod
    def from_mixture_spec(
        cls: type[T_LeRobotMixtureDataset],
        mixture_spec: Sequence[MixtureSpecElement | dict],
        dataset_class: type[LeRobotSingleDataset] | str,
        all_modality_configs: dict[str, dict[str, ModalityConfig]],
        all_transforms: dict[str, ComposedModalityTransform],
        dataset_kwargs: dict | None = None,
        mixture_kwargs: dict | None = None,
    ) -> T_LeRobotMixtureDataset:
        """Initialize the mixture dataset from a specification.

        Args:
            mixture_spec (Sequence[MixtureSpecElement | dict]): The specification for the mixture dataset.
            dataset_class (type[LeRobotSingleDataset] | str): The dataset class or its string path.
            all_modality_configs (dict[str, dict[str, ModalityConfig]]): The modality configs for each embodiment.
            all_transforms (dict[str, ComposedModalityTransform]): The transforms for each embodiment.
            dataset_kwargs (dict | None): Additional keyword arguments for the dataset classes.
            mixture_kwargs (dict | None): Additional keyword arguments for the mixture dataset.

        Returns:
            LeRobotMixtureDataset: The initialized mixture dataset.
        """
        if isinstance(dataset_class, str):
            module_name, class_name = dataset_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
        assert not isinstance(dataset_class, str), f"{dataset_class} is a string"
        assert issubclass(dataset_class, LeRobotSingleDataset), (
            f"{dataset_class} is not a subclass of LeRobotSingleDataset"
        )
        data_mixture = []

        for dataset_spec in tqdm(
            mixture_spec,
            total=len(mixture_spec),
            desc="Initializing datasets",
        ):
            start_time = time.time()
            if isinstance(dataset_spec, dict):
                dataset_spec = MixtureSpecElement.model_validate(dataset_spec)
            datasets = []
            for embodiment_tag, paths in dataset_spec.dataset_path.items():
                if isinstance(paths, Path):
                    paths = [paths]
                for dataset_path in paths:
                    assert embodiment_tag in all_modality_configs, (
                        f"{embodiment_tag} not in modality_configs: {all_modality_configs.keys()}"
                    )
                    assert embodiment_tag in all_transforms, (
                        f"{embodiment_tag} not in transforms: {all_transforms.keys()}"
                    )
                    dataset = dataset_class(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=copy.copy(
                            all_modality_configs[embodiment_tag]
                        ),
                        transforms=copy.copy(all_transforms[embodiment_tag]),
                        **(dataset_kwargs if dataset_kwargs is not None else {}),
                    )
                    datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                if dataset_spec.distribute_weights:
                    weight = relative_length * dataset_spec.dataset_weight
                else:
                    weight = dataset_spec.dataset_weight
                data_mixture.append((dataset, weight))

            print(
                f"Time taken to initialize {len(datasets)} datasets: {time.time() - start_time:.2f} seconds"
            )

        return cls(
            data_mixture=data_mixture,
            **(mixture_kwargs if mixture_kwargs is not None else {}),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the mixture dataset.

        Args:
            index (int): The index to sample (used for deterministic sampling).

        Returns:
            tuple[LeRobotSingleDataset, int, int]: A tuple of (dataset, trajectory_id, step_index).
        """

        # Set seed
        if self.training:
            seed = safe_hash((self.epoch, index, self.seed))
            rng = np.random.default_rng(seed)

            # Sample dataset
            dataset_index = rng.choice(
                len(self.datasets), p=self.dataset_sampling_weights
            )
            dataset = self.datasets[dataset_index]

            if self.allow_padding_at_end:
                # Sample trajectory
                trajectory_index = rng.choice(
                    len(dataset.trajectory_ids),
                    p=self.trajectory_sampling_weights[dataset_index],
                )
                trajectory_id = dataset.trajectory_ids[trajectory_index]

                allowed_length = dataset.trajectory_lengths[trajectory_index]
            else:
                # Avoid padding at the end of the trajectory
                max_delta_index = dataset.max_delta_index
                trajectory_length = 0
                trajectory_id = None
                while trajectory_length < max_delta_index + 1:
                    # Sample trajectory
                    trajectory_index = rng.choice(
                        len(dataset.trajectory_ids),
                        p=self.trajectory_sampling_weights[dataset_index],
                    )
                    trajectory_id = dataset.trajectory_ids[trajectory_index]
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                assert trajectory_id is not None

                # Sample step
                assert trajectory_length >= max_delta_index + 1, (
                    f"{trajectory_length=}, {max_delta_index=}"
                )
                allowed_length = trajectory_length - max_delta_index
            # Get the allowed indices from the step filter
            allowed_indices = dataset.step_filter[trajectory_id]
            # Remove indices that are too large
            allowed_indices = allowed_indices[allowed_indices <= allowed_length]
            step_index = rng.choice(allowed_indices)
            return dataset, trajectory_id, step_index
        else:
            length_cumsum = np.cumsum(self.dataset_lengths)
            dataset_index = np.searchsorted(length_cumsum, index)
            dataset = self.datasets[dataset_index]
            assert (
                len(dataset._lerobot_info_meta.get("discarded_episode_indices", []))
                == 0
            ), (
                f"Find discarded episode indices in evaluation dataset {dataset.dataset_path}"
            )
            trajectory_id, step_index = dataset.all_steps[
                index - length_cumsum[dataset_index]
            ]
            return dataset, trajectory_id, step_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        dataset, trajectory_id, step_index = self.sample_step(index)
        indices = {
            key: delta_indices + step_index
            for key, delta_indices in dataset.delta_indices.items()
        }
        return dataset.transforms(dataset.get_step_data(trajectory_id, indices))

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self.training:
            return int((self.dataset_lengths * self.dataset_sampling_weights).sum())
        else:
            return int(self.dataset_lengths.sum())

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "action.nav_delta": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: Either "weighted_average" or "min_max".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Check if stats are per-horizon (2D) by examining the first task's mean
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            is_per_horizon = first_mean.ndim == 2  # Shape (horizon_len, action_dim)

            if is_per_horizon:
                # Handle per-horizon stats (2D arrays)
                stats_shape = first_mean.shape  # (horizon_len, action_dim)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(stats_shape)
                weighted_squares = np.zeros(stats_shape)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(np.array(stats["min"]))
                    max_list.append(np.array(stats["max"]))
                    q01_list.append(np.array(stats["q01"]))
                    q99_list.append(np.array(stats["q99"]))

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                # Stack along new axis: (num_tasks, horizon_len, action_dim)
                overall_min = np.min(np.stack(min_list, axis=0), axis=0).tolist()
                overall_max = np.max(np.stack(max_list, axis=0), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                q01_array = np.stack(
                    q01_list, axis=0
                )  # (num_tasks, horizon_len, action_dim)
                q99_array = np.stack(q99_list, axis=0)
                if percentile_mixing_method == "weighted_average":
                    # Weighted average along task axis
                    weighted_q01 = np.average(
                        q01_array, axis=0, weights=normalized_weights
                    ).tolist()
                    weighted_q99 = np.average(
                        q99_array, axis=0, weights=normalized_weights
                    ).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(
                        f"Invalid percentile mixing method: {percentile_mixing_method}"
                    )
            else:
                # Handle regular stats (1D arrays)
                num_dims = len(first_mean)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(num_dims)
                weighted_squares = np.zeros(num_dims)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(stats["min"])
                    max_list.append(stats["max"])
                    q01_list.append(stats["q01"])
                    q99_list.append(stats["q99"])

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                overall_min = np.min(np.array(min_list), axis=0).tolist()
                overall_max = np.max(np.array(max_list), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                # Use weighted average of per-task quantiles
                q01_array = np.array(q01_list)
                q99_array = np.array(q99_list)
                if percentile_mixing_method == "weighted_average":
                    weighted_q01 = np.average(
                        q01_array, axis=0, weights=normalized_weights
                    ).tolist()
                    weighted_q99 = np.average(
                        q99_array, axis=0, weights=normalized_weights
                    ).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(
                        f"Invalid percentile mixing method: {percentile_mixing_method}"
                    )

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag
            for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert len(configs) == 1, (
                f"Multiple modality configs for modality {modality}: {list(configs)}"
            )
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag.value not in all_metadatas:
                all_metadatas[dataset.tag.value] = []
            all_metadatas[dataset.tag.value].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag.value])
