"""
api.py
======
FastAPI REST service exposing the clustering pipeline.

Run with:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import settings
from detector import FaceDetector
from embedding import EmbeddingExtractor
from main import FaceClusteringPipeline
from utils import load_image, logger

app = FastAPI(
    title="Face Clustering API",
    description="Cluster unlabeled face images by identity using ArcFace + DBSCAN.",
    version="1.0.0",
)

# Lazily-initialized shared model instances (loaded once, reused across requests)
_detector: FaceDetector | None = None
_embedder: EmbeddingExtractor | None = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def get_embedder() -> EmbeddingExtractor:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingExtractor(detector=get_detector())
    return _embedder


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClusterRequest(BaseModel):
    eps: float = settings.clustering.eps
    min_samples: int = settings.clustering.min_samples


class PredictResponse(BaseModel):
    faces_detected: int
    embeddings: list[list[float]]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(status="ok", model_loaded=_detector is not None)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Detect faces and return their raw ArcFace embeddings for a single image."""
    tmp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)

        image = load_image(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

        faces = get_detector().detect(image, source_name=file.filename)
        if not faces:
            return PredictResponse(faces_detected=0, embeddings=[])

        get_embedder().extract_for_faces(faces)
        embeddings = [f.embedding.tolist() for f in faces]
        return PredictResponse(faces_detected=len(faces), embeddings=embeddings)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/cluster", tags=["Inference"])
async def cluster(files: List[UploadFile] = File(...), eps: float = settings.clustering.eps,
                   min_samples: int = settings.clustering.min_samples) -> JSONResponse:
    """
    Upload a batch of images, run the full clustering pipeline, and return
    a JSON summary (image -> cluster_id, confidence).
    """
    work_dir = Path(tempfile.mkdtemp(prefix="api_cluster_"))
    dataset_dir = work_dir / "dataset"
    output_dir = work_dir / "output"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        for f in files:
            contents = await f.read()
            (dataset_dir / f.filename).write_bytes(contents)

        pipeline = FaceClusteringPipeline(dataset_dir=dataset_dir, output_dir=output_dir, visualize=False)
        pipeline.clusterer.eps = eps
        pipeline.clusterer.min_samples = min_samples
        pipeline.run()

        csv_path = settings.paths.results_csv
        if not csv_path.exists():
            raise HTTPException(status_code=422, detail="No faces detected in the provided images.")

        df = pd.read_csv(csv_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/clusters", tags=["Results"])
def get_clusters() -> JSONResponse:
    """Return the most recent results.csv as JSON, grouped by cluster."""
    csv_path = settings.paths.results_csv
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No results available yet. Run /cluster first.")

    df = pd.read_csv(csv_path)
    grouped = {
        str(cid): group.to_dict(orient="records")
        for cid, group in df.groupby("cluster_id")
    }
    return JSONResponse(content=grouped)


@app.get("/statistics", tags=["Results"])
def get_statistics() -> JSONResponse:
    """Return summary statistics for the most recent clustering run."""
    csv_path = settings.paths.results_csv
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No results available yet. Run /cluster first.")

    df = pd.read_csv(csv_path)
    stats = {
        "total_faces": len(df),
        "n_clusters": int(df.loc[df["cluster_id"] != -1, "cluster_id"].nunique()),
        "n_noise": int((df["cluster_id"] == -1).sum()),
        "mean_confidence": float(df["confidence"].mean()),
        "min_confidence": float(df["confidence"].min()),
        "max_confidence": float(df["confidence"].max()),
    }
    return JSONResponse(content=stats)
