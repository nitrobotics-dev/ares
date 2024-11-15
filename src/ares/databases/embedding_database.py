"""
We want a different embedding index for each robot,and for each robot's state and action spaces.
It would be great to have a unified embedding space of 3D positions, but that relies on finding the forward kinematics model for each robot.
We leave that for future work.

Right now, for each robot, for each episode, we get (N, S) arrays for states and actions.
The embedding indexes only work on flat vectors, so our goal is to normalize and flatten the matrices into vectors.
    - TODO: normalize each sensor range independently
    - TODO: better time dilation? right now just scale all to T timesteps
    - TODO: how to handle over time?

# total 2 indices per robot plus 2 for video, text description (plus one for 3D pos!)
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


class BaseIndexManager(ABC):
    def __init__(self, base_dir: str, max_backups: int = 1, load_existing: bool = True):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata: Dict[str, dict] = {}
        self.max_backups = max_backups
        self.indices: Dict[str, Any] = {}

        # Load existing manager state if it exists
        if load_existing:
            self.load_manager()

    def save_manager(self) -> None:
        """Save the entire state of the index manager"""
        # Save metadata
        metadata_path = self.base_dir / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save each index
        for name in self.indices:
            self.backup_index(name)

    def load_manager(self) -> None:
        """Load the entire state of the index manager"""
        # Load metadata if it exists
        metadata_path = self.base_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r") as f:
                self.metadata = json.load(f)

        # Load indices from directories
        if self.base_dir.exists():
            for robot_dir in self.base_dir.iterdir():
                if robot_dir.is_dir():
                    name = robot_dir.name
                    if self.load_latest_index(name):
                        print(f"Loaded existing index for {name}")

    @abstractmethod
    def init_index(self, name: str, dimension: int) -> None:
        """Initialize a new index for a robot's state or action"""

    @abstractmethod
    def add_matrix(self, name: str, matrix: np.ndarray) -> None:
        """Add a new matrix to robot's index"""

    def _matrix_to_vector(
        self, matrix: np.ndarray, normalize: bool = True, **kwargs: Any
    ) -> np.ndarray:
        """Convert a matrix to (1, N) and optionally normalize

        Args:
            matrix: Input matrix to vectorize
            normalize: Whether to normalize the vector
        """
        vector = matrix.reshape(1, -1).astype(np.float32)
        # TODO: normalize?
        return vector

    @abstractmethod
    def search(
        self, name: str, query_matrix: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar matrices"""

    @abstractmethod
    def backup_index(self, name: str, force: bool = False) -> None:
        """Backup the index for a robot"""

    @abstractmethod
    def load_latest_index(self, name: str) -> bool:
        """Load the most recent index for a robot"""

    def update_metadata(self, name: str, **kwargs: Any) -> None:
        """Update metadata for a robot"""
        if name not in self.metadata:
            self.metadata[name] = {
                "created_at": datetime.now().isoformat(),
                "n_entries": 0,
                "last_backup": None,
            }
        self.metadata[name].update(kwargs)

    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics about the index"""
        if name not in self.metadata:
            raise KeyError(f"No index exists for {name}")
        return {
            "n_entries": self.metadata[name]["n_entries"],
            "created_at": self.metadata[name]["created_at"],
            "last_backup": self.metadata[name]["last_backup"],
        }

    def cleanup_old_backups(self, name: str) -> None:
        """Remove old backups keeping only max_backups most recent"""
        robot_dir = self.base_dir / name
        if not robot_dir.exists():
            return

        # Get all backup files sorted by modification time
        index_files = sorted(
            robot_dir.glob("index_*.faiss"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Keep only max_backups most recent
        for idx_file in index_files[self.max_backups :]:
            timestamp = idx_file.stem[6:]
            idx_file.unlink()
            meta_file = robot_dir / f"meta_{timestamp}.json"
            if meta_file.exists():
                meta_file.unlink()

    @abstractmethod
    def get_all_vectors(self, name: str) -> np.ndarray:
        """Get all vectors stored in the index.

        Args:
            name: Name of the index to query

        Returns:
            np.ndarray: Array of shape (n_entries, dimension) containing all vectors
        """


class FaissIndexManager(BaseIndexManager):
    def __init__(self, base_dir: str, max_backups: int = 1):
        super().__init__(base_dir, max_backups)

    def init_index(self, name: str, dimension: int) -> None:
        """Initialize a new index with specified dimension"""
        if name in self.indices:
            raise ValueError(f"Index for {name} already exists")

        self.indices[name] = faiss.IndexFlatL2(dimension)
        self.update_metadata(name, dimension=dimension)

    def add_matrix(self, name: str, matrix: np.ndarray) -> None:
        """Add a matrix to the index, initializing if needed"""
        if name not in self.indices:
            self.init_index(name, dimension=int(np.prod(matrix.shape)))

        vector = self._matrix_to_vector(matrix)  # , normalize=True)
        self.indices[name].add(vector)
        self.metadata[name]["n_entries"] += 1

    def remove_vectors(self, name: str, indices: List[int]) -> None:
        """Remove vectors at specified indices"""
        if name not in self.indices:
            raise KeyError(f"No index exists for {name}")
        # Note: Basic FAISS indices don't support removal
        # Would need to use IDMap or similar for this functionality
        raise NotImplementedError("Vector removal not supported for basic FAISS index")

    def search(
        self, name: str, query_matrix: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if name not in self.indices:
            raise KeyError(f"No index exists for robot {name}")

        query_vector = self._matrix_to_vector(query_matrix)
        distances, indices = self.indices[name].search(query_vector, k)
        return distances, indices

    def backup_index(self, name: str, force: bool = False) -> None:
        if name not in self.indices:
            return

        robot_dir = self.base_dir / name
        robot_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_path = robot_dir / f"index_{timestamp}.faiss"
        meta_path = robot_dir / f"meta_{timestamp}.json"

        faiss.write_index(self.indices[name], str(index_path))
        self.metadata[name]["last_backup"] = timestamp

        with meta_path.open("w") as f:
            json.dump(self.metadata[name], f, indent=2)

    def load_latest_index(self, name: str) -> bool:
        robot_dir = self.base_dir / name
        if not robot_dir.exists():
            return False

        index_files = list(robot_dir.glob("index_*.faiss"))
        if not index_files:
            return False

        latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
        latest_meta = robot_dir / f"meta_{latest_index.stem[6:]}.json"

        self.indices[name] = faiss.read_index(str(latest_index))
        if latest_meta.exists():
            with latest_meta.open() as f:
                self.metadata[name] = json.load(f)

        return True

    def get_all_vectors(self, name: str) -> np.ndarray:
        """Get all vectors stored in the index.

        Args:
            name: Name of the index to query

        Returns:
            np.ndarray: Array of shape (n_entries, dimension) containing all vectors
        """
        if name not in self.indices:
            raise KeyError(f"No index exists for {name}")

        index = self.indices[name]
        n_vectors = index.ntotal
        return np.vstack([index.reconstruct(i) for i in range(n_vectors)])


# class QdrantIndexManager(BaseIndexManager):
#     def __init__(self, base_dir: str = "robot_indices"):
#         super().__init__(base_dir)
#         # Initialize Qdrant client here

#     def init_robot(self, name: str) -> None:
#         # Create collection for robot
#         pass

#     def add_matrix(self, name: str, matrix: np.ndarray) -> None:
#         # Add points to Qdrant collection
#         pass

#     def search(
#         self, name: str, query_matrix: np.ndarray, k: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         pass

#     def backup_index(self, name: str, force: bool = False) -> None:
#         # Snapshot collection or backup as needed
#         pass

#     def load_latest_index(self, name: str) -> bool:
#         # Load collection from backup
#         pass


def setup_test_index_manager(
    base_dir: str, n_dim: int, n_time: int, n_entries: int
) -> FaissIndexManager:
    manager = FaissIndexManager(base_dir)

    # Initialize indices for different robots and their state/action spaces
    for robot in ["panda", "ur5"]:
        for content in ["state", "action"]:
            name = f"{robot}_{content}"
            manager.init_index(name, dimension=n_dim)

            # Generate some random test data
            for _ in range(n_entries):
                # Simulate episode data: (timesteps, features)
                fake_data = np.random.randn(n_time, n_dim).astype(np.float32)
                manager.add_matrix(name, fake_data)

            # Backup the index
            manager.backup_index(name)
    return manager


if __name__ == "__main__":
    # Create a test index manager
    index_path = "/tmp/robot_indices"
    n_dim = 32
    n_time = 100
    n_entries = 50

    manager = setup_test_index_manager(index_path, n_dim, n_time, n_entries)
    # test loading from disk
    new_manager = FaissIndexManager(index_path)
    breakpoint()

    # Demonstrate search
    name = "panda_state"
    query = np.random.randn(50, 32).astype(np.float32)
    distances, indices = manager.search(name, query, k=5)

    print(f"\nResults for {name}:")
    print(f"Found {len(indices[0])} nearest neighbors")
    print(f"Distances: {distances[0]}")
    print(f"Indices: {indices[0]}")

    # Print stats
    stats = manager.get_stats(name)
    print(f"\nIndex stats for {name}:")
    print(json.dumps(stats, indent=2))

    # Save everything
    manager.save_manager()

    # Create new manager that auto-loads everything
    new_manager = FaissIndexManager("/tmp/robot_indices")
