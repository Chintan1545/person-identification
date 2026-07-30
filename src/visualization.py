"""
visualization.py
=================
Reporting / plotting utilities, deliberately separated from main.py so
the core pipeline can run headless (e.g. inside a Docker API container)
without pulling in matplotlib.

Provides:
  - per-cluster image collages
  - 2D embedding projections (t-SNE and UMAP)
  - summary statistics bar charts
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend - never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import logger


def make_cluster_collage(image_paths: list[Path], out_path: Path, thumb_size: int = 128, max_images: int = 25) -> None:
    """
    Build a grid collage of up to `max_images` thumbnails from one cluster,
    saved as a single JPEG. Useful for quickly eyeballing "does this cluster
    actually look like one person?" without opening every file.
    """
    paths = image_paths[:max_images]
    if not paths:
        logger.warning("No images provided for collage %s", out_path)
        return

    cols = int(np.ceil(np.sqrt(len(paths))))
    rows = int(np.ceil(len(paths) / cols))
    grid = np.full((rows * thumb_size, cols * thumb_size, 3), 255, dtype=np.uint8)

    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_size, thumb_size))
        r, c = divmod(i, cols)
        grid[r * thumb_size:(r + 1) * thumb_size, c * thumb_size:(c + 1) * thumb_size] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    logger.info("Saved collage: %s", out_path)


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method: str = "tsne",
) -> None:
    """
    Project 512-d embeddings down to 2D for visual sanity-checking of
    cluster separation. `method` is "tsne" or "umap".
    """
    if len(embeddings) < 2:
        logger.warning("Not enough embeddings to project (%d).", len(embeddings))
        return

    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        perplexity = max(2, min(30, len(embeddings) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")

    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = "lightgray" if label == -1 else cmap(i % 20)
        name = "noise" if label == -1 else f"cluster {label}"
        plt.scatter(coords[mask, 0], coords[mask, 1], label=name, s=25, color=color, alpha=0.8)

    plt.title(f"Face embedding clusters ({method.upper()} projection)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved embedding projection: %s", out_path)


def plot_cluster_statistics(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: number of images per cluster, plus a confidence histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["cluster_id"].value_counts().sort_index()
    labels = ["noise" if c == -1 else str(c) for c in counts.index]
    axes[0].bar(labels, counts.values, color="steelblue")
    axes[0].set_title("Images per cluster")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].hist(df["confidence"], bins=20, color="indianred", edgecolor="black")
    axes[1].set_title("Confidence score distribution")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved cluster statistics chart: %s", out_path)


def cluster_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-cluster summary DataFrame: image count, mean/min confidence."""
    summary = (
        df.groupby("cluster_id")
        .agg(
            image_count=("image_name", "count"),
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    return summary
