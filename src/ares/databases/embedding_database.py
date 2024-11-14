"""
We want a different embedding index for each robot and for each robot's state and action spaces. It would be great to have a unified embedding space
of 3D positions, but that relies on finding the forward kinematics model for each robot. We leave that for future work. 

Right now, for each robot, for each episode, we get (N, S) arrays for states and actions. The embedding indexes only work on flat vectors, so our goal
is to normalize and flatten the matrices into vectors. 
    - TODO: normalize each sensor range independently
    - TODO: better time dilation? right now just scale all to T timesteps
    - TODO: how to handle over time?
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


class BaseIndexManager(ABC):
    def __init__(self, base_dir: str, max_backups: int = 2):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata: Dict[str, dict] = {}
        self.max_backups = max_backups

    @abstractmethod
    def init_index(self, name: str) -> None:
        """Initialize a new index for a robot's state or action"""
        pass

    @abstractmethod
    def add_matrix(self, name: str, matrix: np.ndarray) -> None:
        """Add a new matrix to robot's index"""
        pass

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
        pass

    @abstractmethod
    def backup_index(self, name: str, force: bool = False) -> None:
        """Backup the index for a robot"""
        pass

    @abstractmethod
    def load_latest_index(self, name: str) -> bool:
        """Load the most recent index for a robot"""
        pass

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


class FaissIndexManager(BaseIndexManager):
    def __init__(self, base_dir: str, max_backups: int = 5):
        super().__init__(base_dir, max_backups)
        self.indices: Dict[str, faiss.IndexFlatL2] = {}

    def init_index(self, name: str, dimension: int) -> None:
        """Initialize a new index with specified dimension"""
        if name in self.indices:
            raise ValueError(f"Index for {name} already exists")

        self.indices[name] = faiss.IndexFlatL2(dimension)
        self.update_metadata(name, dimension=dimension)

    def add_matrix(self, name: str, matrix: np.ndarray) -> None:
        """Add a matrix to the index, initializing if needed"""
        if name not in self.indices:
            self.init_index(name, dimension=np.prod(matrix.shape))

        vector = self._matrix_to_vector(matrix, normalize=True)
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

if __name__ == "__main__":
    # Create a test index manager
    manager = FaissIndexManager("/tmp/robot_indices")

    # Initialize indices for different robots and their state/action spaces
    for robot in ["panda", "ur5"]:
        for content in ["state", "action"]:
            name = f"{robot}_{content}"
            manager.init_index(name, dimension=32)

            # Generate some random test data
            for _ in range(10):
                # Simulate episode data: (timesteps, features)
                fake_data = np.random.randn(50, 32).astype(np.float32)
                manager.add_matrix(name, fake_data)

            # Backup the index
            manager.backup_index(name)

            # Demonstrate search
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
