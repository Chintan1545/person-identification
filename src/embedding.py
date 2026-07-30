"""
embedding.py
============
Converts aligned face crops into 512-dimensional ArcFace embeddings.

Why 512-d?
  The ArcFace backbone (ResNet-based) used in InsightFace's buffalo_l pack
  outputs a 512-length feature vector per face. This is a fixed
  architectural choice of the pretrained model, not something we tune.

Why L2-normalize?
  ArcFace is trained with an angular margin loss, meaning identity is
  encoded in the DIRECTION of the vector, not its magnitude. Normalizing
  every embedding to unit length means cosine similarity reduces to a
  simple dot product, and Euclidean distance becomes a monotonic function
  of cosine similarity - this is what makes cosine-distance clustering
  (DBSCAN, metric="cosine") correct and cheap.

Cosine similarity, in one line:
  sim(a, b) = (a . b) / (||a|| * ||b||)
  For unit vectors, ||a|| = ||b|| = 1, so sim(a, b) = a . b
  Range: -1 (opposite) to 1 (identical direction). Same person -> close to 1.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from config import settings
from detector import DetectedFace, FaceDetector
from utils import logger


class EmbeddingExtractor:
    """
    Wraps ArcFace embedding extraction, reusing the SAME FaceAnalysis
    instance as FaceDetector so the recognition ONNX graph is loaded once.

    Includes an optional on-disk cache keyed by a hash of the image bytes +
    face bbox, so re-running the pipeline (e.g. while tuning DBSCAN eps)
    skips the GPU-heavy embedding step entirely on unchanged inputs.
    """

    def __init__(self, detector: FaceDetector, use_cache: bool = settings.embedding.use_cache) -> None:
        self._detector = detector
        self._use_cache = use_cache
        self._cache_dir = settings.paths.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, aligned_face: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        """
        Return a normalized 512-d embedding for a single ALIGNED 112x112
        face crop. `cache_key` (e.g. "imagename_face0") enables disk caching.
        """
        if self._use_cache and cache_key is not None:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        embedding = self._infer(aligned_face)

        if self._use_cache and cache_key is not None:
            self._save_to_cache(cache_key, embedding)

        return embedding

    def extract_for_faces(self, faces: list[DetectedFace], cache_prefix: str = "") -> list[DetectedFace]:
        """Batch convenience method: fills in `.embedding` on each DetectedFace in place."""
        for i, face in enumerate(faces):
            key = f"{cache_prefix}_face{i}" if cache_prefix else None
            face.embedding = self.extract(face.aligned_face, cache_key=key)
        return faces

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _infer(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Run the ArcFace recognition model on a single aligned crop.

        InsightFace's `FaceAnalysis.models['recognition']` exposes a
        `.get_feat()` method that accepts an already-aligned BGR image
        and returns the raw (unnormalized) embedding.
        """
        rec_model = self._detector.app.models.get("recognition")
        if rec_model is None:
            raise RuntimeError(
                "Recognition model not found in FaceAnalysis pack. "
                "Ensure model_pack includes ArcFace (e.g. 'buffalo_l')."
            )

        raw_embedding = rec_model.get_feat(aligned_face).flatten().astype(np.float32)

        if settings.embedding.normalize:
            return self._l2_normalize(raw_embedding)
        return raw_embedding

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Scale a vector to unit length: v / ||v||. Guards against division by zero."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            logger.warning("Near-zero norm embedding encountered; returning as-is.")
            return vector
        return vector / norm

    # ------------------------------------------------------------------ #
    # Disk cache
    # ------------------------------------------------------------------ #
    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.pkl"

    def _load_from_cache(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read embedding cache %s: %s", path, exc)
            return None

    def _save_to_cache(self, key: str, embedding: np.ndarray) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write embedding cache %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Standalone similarity helpers (used by confidence.py and tests)
# --------------------------------------------------------------------------- #
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Assumes not necessarily normalized."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity. This is the metric DBSCAN clusters on."""
    return 1.0 - cosine_similarity(a, b)
