"""
FAISS-backed and Fallback Semantic Memory implementation for OpenContinualEnv.
"""

import os
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
from open_continual_env.memory.interface import SemanticMemory
from open_continual_env.trajectory.schema import Trajectory

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    ST_AVAILABLE = False


class SimpleEmbedder:
    """Fallback embedder when sentence-transformers is not available."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []
        for t in texts:
            # Deterministic bag-of-words character/hash projection into embedding space
            vec = np.zeros(self.dim, dtype=np.float32)
            words = t.lower().split()
            if not words:
                embeddings.append(vec)
                continue
            for word in words:
                h = sum(ord(c) for c in word)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        arr = np.array(embeddings, dtype=np.float32)
        return arr[0] if isinstance(text, str) else arr


class FAISSMemory(SemanticMemory):
    """
    FAISS-backed semantic memory with graceful fallback to NumPy cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "flat",
    ):
        self.dimension = dimension
        self.trajectories: List[Trajectory] = []
        self.embeddings: List[np.ndarray] = []

        # Default to deterministic 100% local SimpleEmbedder unless explicitly enabled
        if ST_AVAILABLE and os.getenv("ENABLE_HF_EMBEDDINGS") == "1":
            try:
                self.embedder = SentenceTransformer(model_name, local_files_only=True)
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            except Exception:
                self.embedder = SimpleEmbedder(dim=self.dimension)
        else:
            self.embedder = SimpleEmbedder(dim=self.dimension)



        self.use_faiss = FAISS_AVAILABLE
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None

    def _get_embedding(self, text: str) -> np.ndarray:
        if hasattr(self.embedder, "encode"):
            emb = self.embedder.encode(text, normalize_embeddings=True)
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            return emb.astype(np.float32)
        return SimpleEmbedder(dim=self.dimension).encode(text)

    def add(
        self,
        trajectory: Union[Trajectory, Dict[str, Any]],
        embedding: Optional[List[float]] = None
    ) -> str:
        if isinstance(trajectory, dict):
            traj_obj = Trajectory.from_dict(trajectory)
        elif isinstance(trajectory, Trajectory):
            traj_obj = trajectory
        else:
            traj_obj = Trajectory(
                trajectory_id=f"traj_{len(self.trajectories)}",
                prompt=str(trajectory),
                model_response="",
            )

        if embedding is not None:
            emb_vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb_vec)
            if norm > 0:
                emb_vec = emb_vec / norm
        else:
            text_to_embed = f"{traj_obj.prompt}\n{traj_obj.generated_code or traj_obj.model_response}"
            emb_vec = self._get_embedding(text_to_embed)

        self.trajectories.append(traj_obj)
        self.embeddings.append(emb_vec)

        if self.use_faiss and self.index is not None:
            self.index.add(np.expand_dims(emb_vec, axis=0))

        return traj_obj.trajectory_id

    def query_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5
    ) -> List[Trajectory]:
        if not self.trajectories:
            return []

        top_k = min(top_k, len(self.trajectories))
        query_vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(np.expand_dims(query_vec, axis=0), top_k)
            result = []
            for idx in indices[0]:
                if 0 <= idx < len(self.trajectories):
                    result.append(self.trajectories[idx])
            return result
        else:
            # Fallback cosine similarity using numpy
            matrix = np.array(self.embeddings, dtype=np.float32)
            scores = np.dot(matrix, query_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [self.trajectories[i] for i in top_indices]

    def query(
        self,
        prompt: str,
        top_k: int = 5
    ) -> List[Trajectory]:
        query_vec = self._get_embedding(prompt)
        return self.query_by_embedding(query_vec.tolist(), top_k=top_k)

    def get_nearest_distance(self, prompt: str) -> float:
        """Returns distance (1.0 - max similarity) to the nearest trajectory."""
        if not self.trajectories:
            return 1.0
        query_vec = self._get_embedding(prompt)
        matrix = np.array(self.embeddings, dtype=np.float32)
        sims = np.dot(matrix, query_vec)
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
        return max(0.0, 1.0 - max_sim)

    def __len__(self) -> int:
        return len(self.trajectories)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        traj_data = [t.to_dict() if hasattr(t, "to_dict") else dict(t) for t in self.trajectories]
        with open(os.path.join(directory, "trajectories.json"), "w", encoding="utf-8") as f:
            json.dump(traj_data, f, indent=2)

        if self.embeddings:
            np.save(os.path.join(directory, "embeddings.npy"), np.array(self.embeddings))

        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "faiss.index"))

    def load(self, directory: str) -> None:
        traj_path = os.path.join(directory, "trajectories.json")
        if os.path.exists(traj_path):
            with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.trajectories = [Trajectory.from_dict(d) for d in data]

        emb_path = os.path.join(directory, "embeddings.npy")
        if os.path.exists(emb_path):
            arr = np.load(emb_path)
            self.embeddings = [row for row in arr]

        faiss_path = os.path.join(directory, "faiss.index")
        if self.use_faiss and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)


class LegacyExperienceStoreWrapper(SemanticMemory):
    """Wrapper around legacy ExperienceStore to conform to SemanticMemory protocol."""

    def __init__(self, store: Any):
        self.store = store

    def add(
        self,
        trajectory: Union[Trajectory, Dict[str, Any]],
        embedding: Optional[List[float]] = None
    ) -> str:
        if hasattr(self.store, "add"):
            self.store.add(trajectory)
        elif hasattr(self.store, "append"):
            self.store.append(trajectory)
        return getattr(trajectory, "trajectory_id", "legacy_traj")

    def query(
        self,
        prompt: str,
        top_k: int = 5
    ) -> List[Trajectory]:
        if hasattr(self.store, "query"):
            try:
                return self.store.query(prompt, top_k)
            except TypeError:
                # Legacy predicate-style query (ExperienceStore.query(filter_fn))
                # cannot serve semantic lookup; degrade to recency below.
                pass
        if hasattr(self.store, "get_recent"):
            return self.store.get_recent(top_k)
        if hasattr(self.store, "get_all"):
            return self.store.get_all()[-top_k:]
        if isinstance(self.store, list):
            return self.store[-top_k:]
        return []

    def query_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 5
    ) -> List[Trajectory]:
        return self.query("", top_k=top_k)

    def __len__(self) -> int:
        if hasattr(self.store, "__len__"):
            return len(self.store)
        return 0

    def save(self, directory: str) -> None:
        if hasattr(self.store, "save"):
            self.store.save(directory)

    def load(self, directory: str) -> None:
        if hasattr(self.store, "load"):
            self.store.load(directory)
