"""
config.py
=========
Centralized configuration for the face-clustering pipeline.

Every tunable value lives here so that:
  1. No magic numbers are scattered across modules.
  2. Values can be overridden via environment variables (12-factor style),
     which matters once this runs inside Docker / CI.
  3. Changing DBSCAN's `eps` or the detector threshold doesn't require
     hunting through five files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads a local .env file if present, no-op otherwise


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    """All filesystem locations the pipeline touches."""

    root: Path = Path(__file__).resolve().parent.parent
    dataset_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    results_csv: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "dataset_dir", self.root / _env_str("DATASET_DIR", "dataset"))
        object.__setattr__(self, "output_dir", self.root / _env_str("OUTPUT_DIR", "output"))
        object.__setattr__(self, "models_dir", self.root / _env_str("MODELS_DIR", "models"))
        object.__setattr__(self, "logs_dir", self.root / _env_str("LOGS_DIR", "logs"))
        object.__setattr__(self, "cache_dir", self.models_dir / "embedding_cache")
        object.__setattr__(self, "results_csv", self.output_dir / "results.csv")

        for p in (self.dataset_dir, self.output_dir, self.models_dir, self.logs_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DetectorConfig:
    """RetinaFace (via InsightFace) settings."""

    model_pack: str = _env_str("INSIGHTFACE_MODEL", "buffalo_l")
    # ctx_id: -1 = CPU, 0 = first GPU
    ctx_id: int = _env_int("INSIGHTFACE_CTX_ID", -1)
    detection_size: tuple[int, int] = (640, 640)
    min_detection_score: float = _env_float("MIN_DETECTION_SCORE", 0.55)
    min_face_pixels: int = _env_int("MIN_FACE_PIXELS", 40)  # reject tiny/false-positive faces


@dataclass(frozen=True)
class EmbeddingConfig:
    """ArcFace embedding settings."""

    embedding_dim: int = 512
    normalize: bool = True
    use_cache: bool = _env_bool("USE_EMBEDDING_CACHE", True)


@dataclass(frozen=True)
class ClusteringConfig:
    """DBSCAN settings."""

    # eps is a COSINE DISTANCE threshold (1 - cosine_similarity).
    # 0.35-0.45 works well empirically for ArcFace embeddings.
    eps: float = _env_float("DBSCAN_EPS", 0.40)
    min_samples: int = _env_int("DBSCAN_MIN_SAMPLES", 2)
    metric: str = "cosine"
    n_jobs: int = _env_int("DBSCAN_N_JOBS", -1)


@dataclass(frozen=True)
class ConfidenceConfig:
    """Confidence-scoring settings."""

    method: str = _env_str("CONFIDENCE_METHOD", "centroid")  # "centroid" | "knn"
    knn_k: int = _env_int("CONFIDENCE_KNN_K", 5)
    noise_confidence: float = 0.0  # score assigned to DBSCAN noise points (-1 label)


@dataclass(frozen=True)
class QualityConfig:
    """Advanced image-quality filters."""

    enable_blur_filter: bool = _env_bool("ENABLE_BLUR_FILTER", True)
    blur_threshold: float = _env_float("BLUR_THRESHOLD", 60.0)  # Laplacian variance
    enable_duplicate_filter: bool = _env_bool("ENABLE_DUPLICATE_FILTER", True)
    duplicate_hash_size: int = 8  # perceptual hash size


@dataclass(frozen=True)
class AppConfig:
    """Top-level config aggregating all sub-configs."""

    paths: Paths = field(default_factory=Paths)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    log_level: str = _env_str("LOG_LEVEL", "INFO")
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    random_seed: int = _env_int("RANDOM_SEED", 42)


# Single shared instance imported everywhere else: `from config import settings`
settings = AppConfig()
