"""
confidence.py
=============
Assigns a 0-100 confidence score to every clustered face, so a human
reviewer knows which groupings to trust and which to double-check.

DBSCAN itself gives no confidence value - it only outputs a hard label.
We derive confidence AFTER clustering, from the geometry of the embedding
space, using one of two interchangeable strategies:

  1. Centroid-based (default)
     confidence = cosine_similarity(embedding, cluster_centroid) * 100
     Fast (O(1) per point after centroids are computed), intuitive:
     "how close is this face to the average of its cluster".

  2. k-Nearest-Neighbor based
     confidence = mean(cosine_similarity(embedding, its k nearest
                  same-cluster neighbors)) * 100
     More robust to non-spherical / elongated clusters, at the cost of
     O(n) similarity computations per point within the cluster.

Noise points (label == -1) get a fixed low confidence (0.0 by default,
see config.ConfidenceConfig.noise_confidence) since DBSCAN explicitly
could not place them anywhere with sufficient density.

Formula recap:
  cosine_similarity(a, b) = (a . b) / (||a|| ||b||)   -> range [-1, 1]
  confidence = similarity * 100                        -> range [-100, 100]
  In practice, same-person similarities for ArcFace embeddings are almost
  always positive (~0.3 to 1.0), so scores land in a meaningful 0-100 band.
"""

from __future__ import annotations

import numpy as np

from config import settings
from embedding import cosine_similarity
from utils import logger


class ConfidenceScorer:
    """Computes per-face confidence scores after clustering."""

    def __init__(
        self,
        method: str = settings.confidence.method,
        knn_k: int = settings.confidence.knn_k,
        noise_confidence: float = settings.confidence.noise_confidence,
    ) -> None:
        if method not in {"centroid", "knn"}:
            raise ValueError(f"Unknown confidence method: {method}")
        self.method = method
        self.knn_k = knn_k
        self.noise_confidence = noise_confidence

    def score_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Compute a confidence score (0-100) for every embedding.
        `centroids` comes from FaceClusterer.cluster_centroids().
        """
        scores = np.zeros(len(embeddings), dtype=np.float32)

        for cluster_id in set(labels):
            member_idx = np.where(labels == cluster_id)[0]

            if cluster_id == -1:
                scores[member_idx] = self.noise_confidence
                continue

            if self.method == "centroid":
                scores[member_idx] = self._score_centroid(embeddings, member_idx, centroids[cluster_id])
            else:
                scores[member_idx] = self._score_knn(embeddings, member_idx)

        logger.info(
            "Confidence scoring complete (method=%s). Mean=%.1f, Min=%.1f, Max=%.1f",
            self.method, scores.mean() if len(scores) else 0, scores.min() if len(scores) else 0,
            scores.max() if len(scores) else 0,
        )
        return scores

    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_centroid(embeddings: np.ndarray, member_idx: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """confidence = cosine_similarity(embedding, cluster centroid) * 100."""
        out = np.zeros(len(member_idx), dtype=np.float32)
        for i, idx in enumerate(member_idx):
            sim = cosine_similarity(embeddings[idx], centroid)
            out[i] = max(0.0, sim) * 100.0  # clip negative similarity to 0 confidence
        return out

    @staticmethod
    def _score_knn(embeddings: np.ndarray, member_idx: np.ndarray, k: int = 5) -> np.ndarray:
        """confidence = mean similarity to the k nearest same-cluster neighbors * 100."""
        out = np.zeros(len(member_idx), dtype=np.float32)
        cluster_embeddings = embeddings[member_idx]

        for i, idx in enumerate(member_idx):
            sims = np.array([cosine_similarity(embeddings[idx], other) for other in cluster_embeddings])
            sims = np.delete(sims, i)  # exclude self-similarity (always 1.0)
            if len(sims) == 0:
                out[i] = 100.0  # only member of its cluster (shouldn't happen with min_samples>=2)
                continue
            top_k = np.sort(sims)[-k:] if len(sims) >= k else sims
            out[i] = max(0.0, float(top_k.mean())) * 100.0

        return out
