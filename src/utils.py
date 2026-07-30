"""
utils.py
========
Shared, stateless helper functions used across the pipeline:
  - logging setup
  - image loading / listing
  - CSV I/O
  - output folder management (copying clustered images)
  - perceptual hashing for duplicate detection
  - blur detection

Keeping these separate from business logic (detector/embedding/clustering)
avoids duplicating I/O code in every module.
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from config import settings


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(name: str = "face_clustering") -> logging.Logger:
    """
    Configure and return a logger that writes to both console (rich-formatted
    if available) and a rotating file under logs/.

    Called once per process; subsequent calls just return the same logger
    because we guard against duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate log lines
        return logger

    logger.setLevel(settings.log_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Rotating file handler (5 MB x 3 backups)
    log_file = settings.paths.logs_dir / f"{name}.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# --------------------------------------------------------------------------- #
# Image discovery / loading
# --------------------------------------------------------------------------- #
def list_images(directory: Path) -> list[Path]:
    """Return all supported image paths under `directory`, sorted for determinism."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in settings.supported_extensions
    )
    logger.info("Found %d candidate images in %s", len(paths), directory)
    return paths


def load_image(path: Path) -> np.ndarray | None:
    """
    Load an image as a BGR numpy array (OpenCV convention).
    Returns None (and logs a warning) instead of raising, so a single
    corrupted file never kills a batch job.
    """
    try:
        # imdecode handles unicode paths better than cv2.imread on some platforms
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this is I/O boundary
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Quality filters
# --------------------------------------------------------------------------- #
def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Blur metric: the variance of the Laplacian. Sharp images have high
    variance (lots of edges); blurry images have low variance.
    Standard, cheap, well-understood heuristic - no ML model needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def average_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Cheap perceptual hash (aHash) used for near-duplicate detection.
    Resizes to a tiny grayscale thumbnail, thresholds against the mean,
    and packs the bits into a hex string. Two visually similar images
    (even after minor re-encoding) produce identical or near-identical hashes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    # pack bits into hex for compact storage/comparison
    bit_string = "".join("1" if b else "0" for b in bits)
    return f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"


# --------------------------------------------------------------------------- #
# CSV / results
# --------------------------------------------------------------------------- #
@dataclass
class FaceRecord:
    """One row of the final results.csv - one row per DETECTED FACE, not per image."""

    image_name: str
    image_path: str
    face_index: int
    cluster_id: int
    confidence: float
    embedding_dim: int
    detection_score: float
    bbox: str  # "x1,y1,x2,y2" - stored as string for flat CSV compatibility
    is_noise: bool


def save_results_csv(records: Iterable[FaceRecord], csv_path: Path) -> None:
    """Write all FaceRecords to results.csv."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]

    if not rows:
        logger.warning("No records to write - results.csv will only contain headers.")
        fieldnames = [f.name for f in FaceRecord.__dataclass_fields__.values()]
    else:
        fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d records to %s", len(rows), csv_path)


def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Read results.csv back as a DataFrame (used by Streamlit / API)."""
    return pd.read_csv(csv_path)


# --------------------------------------------------------------------------- #
# Output folder management
# --------------------------------------------------------------------------- #
def create_cluster_folders(records: Iterable[FaceRecord], output_dir: Path) -> None:
    """
    Copy each source image into output/cluster_<id>/ (or output/noise/ for
    unclustered faces). Copies rather than moves, so the original dataset
    is never mutated.
    """
    output_dir = Path(output_dir)

    for record in records:
        folder_name = "noise" if record.is_noise else f"cluster_{record.cluster_id}"
        target_dir = output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        src = Path(record.image_path)
        # face_index disambiguates multiple faces detected in the same source image
        dst_name = f"{src.stem}_face{record.face_index}{src.suffix}"
        dst = target_dir / dst_name

        try:
            shutil.copy2(src, dst)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to copy %s -> %s: %s", src, dst, exc)

    logger.info("Cluster folders written to %s", output_dir)


def clean_output_dir(output_dir: Path) -> None:
    """Wipe the output directory before a fresh run, so stale clusters don't linger."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cleaned output directory: %s", output_dir)
