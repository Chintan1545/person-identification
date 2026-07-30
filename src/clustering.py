"""
clustering.py
=============
Groups face embeddings into per-person clusters using DBSCAN.

Why DBSCAN and not the alternatives?

  KMeans
    Requires K (number of clusters) up front. We don't know how many
    people are in the dataset - that's the whole point of the problem.
    KMeans also assumes roughly spherical, equally-sized clusters and
    forces EVERY point into a cluster, so a single stranger's photo
    would get incorrectly merged into someone else's group.

  Agglomerative Clustering
    Can work without knowing K if you cut the dendrogram at a distance
    threshold (similar idea to DBSCAN's eps), but it still assigns every
    point to some cluster - no native concept of "noise" - and its
    complexity is more sensitive to bad merges early in the tree.

  Spectral Clustering
    Needs K specified, and scales poorly (eigendecomposition of an NxN
    similarity matrix) - impractical once you get into the thousands of
    faces this project is meant to scale toward.

  DBSCAN
    - Does NOT require K.
    - Has a native "noise" label (-1) for points that don't belong
      confidently to any group - exactly what we want for a stray face,
      a bad crop, or a person who appears only once.
    - Density-based, so it naturally handles clusters of very different
      sizes (one person with 40 photos, another with 2).
    - Works directly on cosine distance, which is the metric ArcFace
      embeddings were trained to be discriminative under.

Key parameters:
  eps (float)
    Maximum cosine distance between two points for them to be considered
    neighbors. Cosine distance = 1 - cosine_similarity, so eps=0.4 means
    "similarity >= 0.6 to be linked". Too low -> over-splits one person
    into many clusters. Too high -> merges different people together.

  min_samples (int)
    Minimum number of neighbors (including itself) required for a point
    to be a "core point" that can start/grow a cluster. min_samples=2
    means even a pair of similar photos can form a valid cluster - useful
    for small per-person photo counts.

  metric="cosine"
    Tells scikit-learn to compute pairwise cosine distance directly
    instead of assuming Euclidean space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from config import settings
from utils import logger


@dataclass
class ClusteringResult:
    labels: np.ndarray            # cluster id per embedding, -1 = noise
    n_clusters: int
    n_noise: int


class FaceClusterer:
    """Thin, testable wrapper around sklearn's DBSCAN for face embeddings."""

    def __init__(
        self,
        eps: float = settings.clustering.eps,
        min_samples: int = settings.clustering.min_samples,
        metric: str = settings.clustering.metric,
        n_jobs: int = settings.clustering.n_jobs,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Cluster an (N, 512) array of embeddings.

        Returns labels where:
          -1           -> noise (unclustered / likely a one-off face)
          0, 1, 2, ...  -> cluster ids, one per identified person
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D array of embeddings, got shape {embeddings.shape}")

        if len(embeddings) == 0:
            logger.warning("No embeddings provided to clusterer.")
            return ClusteringResult(labels=np.array([]), n_clusters=0, n_noise=0)

        logger.info(
            "Running DBSCAN on %d embeddings (eps=%.2f, min_samples=%d, metric=%s)",
            len(embeddings), self.eps, self.min_samples, self.metric,
        )

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        labels = model.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        logger.info("DBSCAN found %d clusters and %d noise points", n_clusters, n_noise)

        return ClusteringResult(labels=labels, n_clusters=n_clusters, n_noise=n_noise)

    def cluster_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
        """
        Compute the mean (then re-normalized) embedding per cluster.
        Used by confidence.py for centroid-based confidence scoring.
        Noise (-1) is excluded - it has no meaningful centroid.
        """
        centroids: dict[int, np.ndarray] = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            members = embeddings[labels == cluster_id]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[cluster_id] = centroid / norm if norm > 1e-10 else centroid
        return centroids
