"""
We want a different embedding index for each robot,and for each robot's state and action spaces.
It would be great to have a unified embedding space of 3D positions, but that relies on finding the forward kinematics model for each robot.
We leave that for future work.

Right now, for each robot, for each episode, we get (N, S) arrays for states and actions.
The embedding indexes only work on flat vectors, so our goal is to normalize and flatten the matrices into vectors.
    - TODO: normalize each sensor range independently
    - TODO: better time dilation? right now just scale all to T timesteps

Total 2 indices per robot plus 2 for video, text description (plus one for 3D pos!)
"""

import json
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import faiss
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ares.configs.base import Rollout

BASE_EMBEDDING_DB_PATH = "/tmp/ares_dump/embedding_data"
TEST_EMBEDDING_DB_PATH = "/tmp/ares_dump/test_embedding_data"


def rollout_to_index_name(rollout: Rollout | pd.Series, suffix: str) -> str:
    if isinstance(rollout, pd.Series):
        return f"{rollout['dataset_name']}-{rollout['robot_embodiment']}-{suffix}"
    return f"{rollout.dataset_name}-{rollout.robot.embodiment}-{suffix}"


def rollout_to_embedding_pack(
    rollout: Rollout | pd.Series, suffix: str
) -> Dict[str, np.ndarray | None]:
    name = rollout_to_index_name(rollout, suffix)
    states = (
        rollout.trajectory.states_array
        if isinstance(rollout, Rollout)
        else rollout["trajectory_states_array"]
    )
    actions = (
        rollout.trajectory.actions_array
        if isinstance(rollout, Rollout)
        else rollout["trajectory_actions_array"]
    )
    return {
        f"{name}-states": states,
        f"{name}-actions": actions,
    }


class Index(ABC):
    """Base class for vector indices"""

    @abstractmethod
    def __init__(self, feature_dim: int, time_steps: int):
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.total_dim = feature_dim * time_steps
        self.n_entries = 0

        # Initialize normalization constants to None
        self.norm_means: Optional[np.ndarray] = None
        self.norm_stds: Optional[np.ndarray] = None

    @abstractmethod
    def add_vector(self, vector: np.ndarray, entry_id: str) -> None:
        """Add a single vector to the index"""
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Search for similar vectors
        Returns:
            - distances: (n_queries, k) array of distances
            - ids: list of string IDs corresponding to the matches
            - vectors: (k, dimension) array of the matched vectors
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk"""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk"""
        pass

    @abstractmethod
    def get_all_vectors(self) -> np.ndarray:
        """Get all vectors in the index"""
        pass

    def set_normalization(self, means: np.ndarray, stds: np.ndarray) -> None:
        """Set normalization constants for each channel"""
        if means.shape[0] != self.feature_dim or stds.shape[0] != self.feature_dim:
            raise ValueError(
                f"Normalization constants must have shape ({self.feature_dim},)"
            )
        self.norm_means = means
        self.norm_stds = stds

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply channel-wise normalization if constants are set"""
        if self.norm_means is None or self.norm_stds is None:
            return matrix

        # Broadcasting will automatically align the dimensions
        return (matrix - self.norm_means) / self.norm_stds

    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Reverse normalization if constants are set"""
        if self.norm_means is None or self.norm_stds is None:
            return matrix

        # Broadcasting will automatically align the dimensions
        return (matrix * self.norm_stds) + self.norm_means


class FaissIndex(Index):
    def __init__(self, feature_dim: int, time_steps: int):
        super().__init__(feature_dim, time_steps)
        base_index = faiss.IndexFlatL2(self.total_dim)
        self.index = faiss.IndexIDMap2(base_index)
        self.id_map: Dict[int, str] = {}
        self.next_id: int = 0

    def add_vector(self, vector: np.ndarray, entry_id: str) -> None:
        internal_id = self.next_id
        self.next_id += 1
        self.id_map[internal_id] = entry_id
        self.index.add_with_ids(vector.reshape(1, -1), np.array([internal_id]))
        self.n_entries += 1

    def search(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        distances, internal_indices = self.index.search(query_vector.reshape(1, -1), k)
        string_ids = [self.id_map[int(idx)] for idx in internal_indices[0] if idx != -1]
        vectors = np.vstack(
            [
                self.index.reconstruct(int(idx))
                for idx in internal_indices[0]
                if idx != -1
            ]
        )
        return distances, string_ids, vectors

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))
        meta = {
            "feature_dim": self.feature_dim,
            "time_steps": self.time_steps,
            "n_entries": self.n_entries,
            "id_map": self.id_map,
            "next_id": self.next_id,
            "norm_means": (
                self.norm_means.tolist() if self.norm_means is not None else None
            ),
            "norm_stds": (
                self.norm_stds.tolist() if self.norm_stds is not None else None
            ),
        }
        with (path.parent / f"{path.stem}_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        meta_path = path.parent / f"{path.stem}_meta.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
                self.feature_dim = meta["feature_dim"]
                self.time_steps = meta["time_steps"]
                self.total_dim = self.feature_dim * self.time_steps
                self.n_entries = meta["n_entries"]
                self.id_map = {int(k): v for k, v in meta["id_map"].items()}
                self.next_id = meta["next_id"]

                if meta["norm_means"] is not None:
                    self.norm_means = np.array(meta["norm_means"])
                    self.norm_stds = np.array(meta["norm_stds"])

    def get_all_vectors(self) -> np.ndarray:
        return np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])


class IndexManager:
    def __init__(self, base_dir: str, index_class: Type[Index], max_backups: int = 1):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.index_class = index_class
        self.max_backups = max_backups
        self.indices: Dict[str, Index] = {}
        self.metadata: Dict[str, dict] = {}

        # Load existing indices if they exist
        self.load()

    def init_index(
        self,
        name: str,
        feature_dim: int,
        time_steps: int,
        norm_means: Optional[np.ndarray] = None,
        norm_stds: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a new index with specified dimensions and optional normalization"""
        if name in self.indices:
            raise ValueError(f"Index {name} already exists")

        index = self.index_class(feature_dim, time_steps)
        if norm_means is not None and norm_stds is not None:
            index.set_normalization(norm_means, norm_stds)

        self.indices[name] = index
        self.update_metadata(
            name,
            feature_dim=feature_dim,
            time_steps=time_steps,
            has_normalization=norm_means is not None,
        )

    def _interpolate_matrix(
        self, matrix: np.ndarray, target_time_steps: int
    ) -> np.ndarray:
        """Interpolate matrix to target number of time steps"""
        current_steps = matrix.shape[0]
        if current_steps == target_time_steps:
            return matrix

        # Create evenly spaced points for interpolation
        current_times = np.linspace(0, 1, current_steps)
        target_times = np.linspace(0, 1, target_time_steps)

        # Interpolate each feature dimension
        interpolated = np.zeros((target_time_steps, matrix.shape[1]))
        for feature in range(matrix.shape[1]):
            interpolator = interp1d(current_times, matrix[:, feature], kind="linear")
            interpolated[:, feature] = interpolator(target_times)

        return interpolated

    def add_matrix(self, name: str, matrix: np.ndarray, entry_id: str) -> None:
        """Add a matrix to the index, applying interpolation and normalization"""
        if name not in self.indices:
            feature_dim = matrix.shape[1]
            default_time_steps = matrix.shape[0]
            self.init_index(
                name, feature_dim=feature_dim, time_steps=default_time_steps
            )

        index = self.indices[name]
        interpolated = self._interpolate_matrix(matrix, index.time_steps)
        normalized = index.normalize_matrix(interpolated)
        vector = normalized.flatten()
        try:
            index.add_vector(vector, entry_id)
        except Exception as e:
            print(f"Error adding vector to index {name}: {e}; {traceback.format_exc()}")
            breakpoint()
        self.metadata[name]["n_entries"] += 1

    def load(self) -> None:
        """Load indices and metadata from disk"""
        # Load manager metadata first
        metadata_path = self.base_dir / "manager_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r") as f:
                self.metadata = json.load(f)

        # Load indices
        for path in self.base_dir.iterdir():
            if path.suffix == ".index":
                name = path.stem
                # Use metadata to get the correct dimensions
                if name in self.metadata:
                    self.init_index(
                        name,
                        feature_dim=self.metadata[name]["feature_dim"],
                        time_steps=self.metadata[name]["time_steps"],
                    )
                    self.load_index(name)

    def load_index(self, name: str) -> None:
        """Load an index from disk"""
        path = self.base_dir / f"{name}.index"
        if path.exists():
            self.indices[name].load(path)

    def save(self) -> None:
        """Save indices and metadata to disk"""
        # Save all indices
        for name, index in self.indices.items():
            self.save_index(name)

        # Save manager metadata
        metadata_path = self.base_dir / "manager_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_index(self, name: str) -> None:
        """Save an index to disk"""
        path = self.base_dir / f"{name}.index"
        self.indices[name].save(path)

    def update_metadata(self, name: str, **kwargs: Any) -> None:
        """Update metadata for an index"""
        if name not in self.metadata:
            self.metadata[name] = {"n_entries": 0}  # Initialize with n_entries
        self.metadata[name].update(kwargs)  # Update with additional metadata

    def get_stats(self, name: str) -> dict:
        """Get statistics for an index"""
        return self.metadata[name]

    def search_matrix(
        self, name: str, query_matrix: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search for similar matrices, handling normalization"""
        index = self.indices[name]
        interpolated = self._interpolate_matrix(query_matrix, index.time_steps)
        normalized = index.normalize_matrix(interpolated)
        query_vector = normalized.flatten()

        distances, ids, vectors = index.search(query_vector, k)
        distances = distances[0]  # Take first row since only one query

        # Reshape and denormalize the results
        matrices = []
        for v in vectors:
            # Direct reshape using the index dimensions
            matrix = v.reshape(index.time_steps, index.feature_dim)
            denormalized = index.denormalize_matrix(matrix)
            matrices.append(denormalized)

        return distances, np.array(ids), np.array(matrices)

    def set_normalization(self, name: str, means: np.ndarray, stds: np.ndarray) -> None:
        """Set normalization constants for an existing index"""
        if name not in self.indices:
            raise ValueError(f"Index {name} does not exist")

        self.indices[name].set_normalization(means, stds)
        self.metadata[name]["has_normalization"] = True

    def get_all_matrices(
        self, name: str | list[str] | None = None
    ) -> Dict[str, np.ndarray]:
        """Get all vectors the manager, reshaping them to matrices. Pass a name or list of names to get a single index's vectors."""
        if not name:
            return {
                name: index.get_all_vectors().reshape(
                    -1, index.time_steps, index.feature_dim
                )
                for name, index in self.indices.items()
            }
        if isinstance(name, str):
            name = [name]

        return {
            n: self.indices[n]
            .get_all_vectors()
            .reshape(-1, self.indices[n].time_steps, self.indices[n].feature_dim)
            for n in name
        }
