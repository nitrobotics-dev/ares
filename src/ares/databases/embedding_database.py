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
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import faiss
import numpy as np
from scipy.interpolate import interp1d


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

        normalized = matrix.copy()
        for i in range(self.feature_dim):
            normalized[:, i] = (matrix[:, i] - self.norm_means[i]) / self.norm_stds[i]
        return normalized

    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Reverse normalization if constants are set"""
        if self.norm_means is None or self.norm_stds is None:
            return matrix

        denormalized = matrix.copy()
        for i in range(self.feature_dim):
            denormalized[:, i] = (matrix[:, i] * self.norm_stds[i]) + self.norm_means[i]
        return denormalized


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
        self.load_manager()

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

        index.add_vector(vector, entry_id)
        self.metadata[name]["n_entries"] += 1

    def load_manager(self) -> None:
        """Load indices and metadata from disk"""
        for path in self.base_dir.iterdir():
            if path.is_file():
                continue
            name = path.stem
            feature_dim = int(np.prod(self.indices[name].get_all_vectors().shape))
            self.init_index(
                name, feature_dim=feature_dim, time_steps=self.indices[name].time_steps
            )
            self.load_index(name)

    def load_index(self, name: str) -> None:
        """Load an index from disk"""
        path = self.base_dir / f"{name}.index"
        if path.exists():
            self.indices[name].load(path)

    def save_manager(self) -> None:
        """Save indices and metadata to disk"""
        for name, index in self.indices.items():
            self.save_index(name)

    def save_index(self, name: str) -> None:
        """Save an index to disk"""
        path = self.base_dir / f"{name}.index"
        self.indices[name].save(path)

    def update_metadata(self, name: str, **kwargs: Any) -> None:
        """Update metadata for an index"""
        self.metadata[name] = kwargs

    def get_stats(self, name: str) -> dict:
        """Get statistics for an index"""
        return self.metadata[name]

    def search_matrix(
        self, name: str, query_matrix: np.ndarray, k: int
    ) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
        """Search for similar matrices, handling normalization"""
        index = self.indices[name]
        interpolated = self._interpolate_matrix(query_matrix, index.time_steps)
        normalized = index.normalize_matrix(interpolated)
        query_vector = normalized.flatten()

        distances, ids, vectors = index.search(query_vector, k)

        # Reshape and denormalize the results
        matrices = []
        for v in vectors:
            matrix = index.reshape_vector(v)
            denormalized = index.denormalize_matrix(matrix)
            matrices.append(denormalized)

        return distances, ids, matrices

    def set_normalization(self, name: str, means: np.ndarray, stds: np.ndarray) -> None:
        """Set normalization constants for an existing index"""
        if name not in self.indices:
            raise ValueError(f"Index {name} does not exist")

        self.indices[name].set_normalization(means, stds)
        self.metadata[name]["has_normalization"] = True


if __name__ == "__main__":
    # Test the index manager with synthetic data
    import shutil

    from scipy.stats import norm

    # Create test directory
    test_dir = Path("test_index_manager")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Initialize manager
    manager = IndexManager(str(test_dir), FaissIndex)

    # Create synthetic data with varying time lengths
    feature_dim = 3
    n_samples = 5

    # Set up normalization constants
    norm_means = np.array([0.0, 1.0, -0.5])
    norm_stds = np.array([1.0, 2.0, 0.5])

    # Generate test matrices of different lengths
    matrices = []
    for i in range(n_samples):
        # Random length between 8-12 timesteps
        time_steps = np.random.randint(8, 13)
        # Create a matrix with some pattern plus noise
        t = np.linspace(0, 2 * np.pi, time_steps)
        matrix = np.zeros((time_steps, feature_dim))
        matrix[:, 0] = np.sin(t) + norm.rvs(size=time_steps, scale=0.1)
        matrix[:, 1] = np.cos(t) + norm.rvs(size=time_steps, scale=0.1)
        matrix[:, 2] = np.sin(2 * t) + norm.rvs(size=time_steps, scale=0.1)
        matrices.append(matrix)

    # Initialize index with normalization
    manager.init_index(
        "test_index",
        feature_dim=feature_dim,
        time_steps=10,  # We'll standardize to 10 time steps
        norm_means=norm_means,
        norm_stds=norm_stds,
    )

    # Add matrices
    for i, matrix in enumerate(matrices):
        manager.add_matrix("test_index", matrix, f"entry_{i}")
        print(f"Added matrix {i} with shape {matrix.shape}")

    # Test search
    query_matrix = matrices[0]  # Use first matrix as query
    print(f"\nSearching with query matrix of shape {query_matrix.shape}")

    distances, ids, result_matrices = manager.search_matrix(
        "test_index", query_matrix, k=3
    )

    print("\nSearch results:")
    print(f"Distances: {distances[0]}")
    print(f"IDs: {ids}")
    print(f"Result matrix shapes: {[m.shape for m in result_matrices]}")

    # Save and reload
    print("\nSaving and reloading index...")
    manager.save_all()

    new_manager = IndexManager(test_dir, FaissIndex)
    distances, ids, result_matrices = new_manager.search_matrix(
        "test_index", query_matrix, k=3
    )

    print("\nSearch results after reload:")
    print(f"Distances: {distances[0]}")
    print(f"IDs: {ids}")
    print(f"Result matrix shapes: {[m.shape for m in result_matrices]}")

    # Verify normalization was preserved
    index = new_manager.indices["test_index"]
    print("\nNormalization constants after reload:")
    print(f"Means: {index.norm_means}")
    print(f"Stds: {index.norm_stds}")

    # Clean up
    shutil.rmtree(test_dir)
