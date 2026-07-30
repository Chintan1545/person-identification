"""
main.py
=======
Pipeline orchestrator. Contains NO business logic of its own - it only
sequences calls to the other modules, exactly as laid out in the
architecture doc:

    load_images -> detect -> (quality filter) -> embed -> cluster
        -> confidence -> save CSV -> save cluster folders -> visualize

Run as:
    python src/main.py --dataset dataset/ --output output/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from clustering import FaceClusterer
from confidence import ConfidenceScorer
from config import settings
from detector import FaceDetector
from embedding import EmbeddingExtractor
from utils import (
    FaceRecord,
    average_hash,
    clean_output_dir,
    create_cluster_folders,
    list_images,
    load_image,
    logger,
    save_results_csv,
    variance_of_laplacian,
)
from visualization import (
    cluster_summary_table,
    make_cluster_collage,
    plot_cluster_statistics,
    plot_embedding_projection,
)


class FaceClusteringPipeline:
    """Coordinates the full clustering pipeline end-to-end."""

    def __init__(self, dataset_dir: Path, output_dir: Path, visualize: bool = True) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.visualize = visualize

        self.detector = FaceDetector()
        self.embedder = EmbeddingExtractor(detector=self.detector)
        self.clusterer = FaceClusterer()
        self.scorer = ConfidenceScorer()

        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        start = time.time()
        clean_output_dir(self.output_dir)

        image_paths = list_images(self.dataset_dir)
        if not image_paths:
            logger.error("No images found in %s - aborting.", self.dataset_dir)
            return

        faces_meta, embeddings = self._detect_and_embed(image_paths)
        if not faces_meta:
            logger.error("No faces detected in any image - aborting.")
            return

        embeddings_arr = np.stack(embeddings)
        result = self.clusterer.fit(embeddings_arr)
        centroids = self.clusterer.cluster_centroids(embeddings_arr, result.labels)
        confidences = self.scorer.score_all(embeddings_arr, result.labels, centroids)

        records = self._build_records(faces_meta, result.labels, confidences)

        save_results_csv(records, settings.paths.results_csv)
        create_cluster_folders(records, self.output_dir)

        if self.visualize:
            self._generate_visuals(records, embeddings_arr, result.labels)

        self._print_summary(records, result, time.time() - start)

    # ------------------------------------------------------------------ #
    def _detect_and_embed(self, image_paths: list[Path]):
        """Stage 1+2+3: load -> detect -> quality-filter -> embed. Returns parallel lists."""
        faces_meta = []   # list of dicts describing each accepted face
        embeddings = []   # list of np.ndarray, same order/index as faces_meta

        for path in tqdm(image_paths, desc="Detecting & embedding", unit="img"):
            image = load_image(path)
            if image is None:
                continue

            if settings.quality.enable_duplicate_filter:
                img_hash = average_hash(image, settings.quality.duplicate_hash_size)
                if img_hash in self._seen_hashes:
                    logger.info("Skipping near-duplicate image: %s", path.name)
                    continue
                self._seen_hashes.add(img_hash)

            if settings.quality.enable_blur_filter:
                blur_score = variance_of_laplacian(image)
                if blur_score < settings.quality.blur_threshold:
                    logger.info("Skipping blurry image (%.1f < %.1f): %s",
                                blur_score, settings.quality.blur_threshold, path.name)
                    continue

            detected_faces = self.detector.detect(image, source_name=path.name)
            if not detected_faces:
                continue

            self.embedder.extract_for_faces(detected_faces, cache_prefix=path.stem)

            for face_idx, face in enumerate(detected_faces):
                faces_meta.append({
                    "image_name": path.name,
                    "image_path": str(path),
                    "face_index": face_idx,
                    "detection_score": face.detection_score,
                    "bbox": face.bbox,
                })
                embeddings.append(face.embedding)

        logger.info("Accepted %d faces across %d source images.", len(faces_meta), len(image_paths))
        return faces_meta, embeddings

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_records(faces_meta: list[dict], labels: np.ndarray, confidences: np.ndarray) -> list[FaceRecord]:
        records = []
        for meta, label, conf in zip(faces_meta, labels, confidences):
            records.append(
                FaceRecord(
                    image_name=meta["image_name"],
                    image_path=meta["image_path"],
                    face_index=meta["face_index"],
                    cluster_id=int(label),
                    confidence=round(float(conf), 2),
                    embedding_dim=settings.embedding.embedding_dim,
                    detection_score=round(float(meta["detection_score"]), 4),
                    bbox=",".join(map(str, meta["bbox"])),
                    is_noise=bool(label == -1),
                )
            )
        return records

    # ------------------------------------------------------------------ #
    def _generate_visuals(self, records: list[FaceRecord], embeddings: np.ndarray, labels: np.ndarray) -> None:
        import pandas as pd

        df = pd.DataFrame([r.__dict__ for r in records])
        viz_dir = self.output_dir / "_visualizations"

        try:
            plot_embedding_projection(embeddings, labels, viz_dir / "tsne_projection.png", method="tsne")
        except Exception as exc:  # noqa: BLE001
            logger.warning("t-SNE projection failed: %s", exc)

        try:
            plot_cluster_statistics(df, viz_dir / "cluster_statistics.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Statistics plot failed: %s", exc)

        for cluster_id in sorted(df["cluster_id"].unique()):
            if cluster_id == -1:
                continue
            paths = [Path(p) for p in df.loc[df["cluster_id"] == cluster_id, "image_path"]]
            make_cluster_collage(paths, viz_dir / f"collage_cluster_{cluster_id}.jpg")

        summary = cluster_summary_table(df)
        summary.to_csv(viz_dir / "cluster_summary.csv", index=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_summary(records: list[FaceRecord], result, elapsed: float) -> None:
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("Total faces processed : %d", len(records))
        logger.info("Clusters found         : %d", result.n_clusters)
        logger.info("Noise (unclustered)    : %d", result.n_noise)
        logger.info("Elapsed time            : %.1fs", elapsed)
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster unlabeled face images by identity.")
    parser.add_argument("--dataset", type=Path, default=settings.paths.dataset_dir, help="Input image folder")
    parser.add_argument("--output", type=Path, default=settings.paths.output_dir, help="Output folder")
    parser.add_argument("--no-visualize", action="store_true", help="Skip chart/collage generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = FaceClusteringPipeline(
        dataset_dir=args.dataset,
        output_dir=args.output,
        visualize=not args.no_visualize,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
