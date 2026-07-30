"""
detector.py
===========
Face detection using InsightFace's RetinaFace model (bundled in the
"buffalo_l" model pack). Responsible ONLY for finding faces and their
landmarks - it does NOT compute embeddings (see embedding.py).

Design notes:
  - The InsightFace `FaceAnalysis` app is expensive to initialize (loads
    ONNX models), so `FaceDetector` loads it once in __init__ and reuses
    it for every image (singleton-per-pipeline-run pattern).
  - Detection failures (no face, corrupted image) never raise; they return
    an empty list so the caller can log-and-skip without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings
from utils import logger


@dataclass
class DetectedFace:
    """Everything the pipeline needs about one detected face."""

    bbox: tuple[int, int, int, int]          # x1, y1, x2, y2 in pixel coords
    landmarks: np.ndarray                     # shape (5, 2) - eyes, nose, mouth corners
    detection_score: float                    # RetinaFace confidence, 0-1
    aligned_face: np.ndarray                   # 112x112 BGR crop, ready for ArcFace
    embedding: np.ndarray | None = None        # filled in later by embedding.py


class FaceDetector:
    """
    Thin, testable wrapper around insightface.app.FaceAnalysis.

    Kept as a class (not free functions) because it owns an expensive,
    stateful resource (the loaded ONNX model) that should be initialized
    once and reused - a textbook case for OOP over procedural code.
    """

    def __init__(
        self,
        model_pack: str = settings.detector.model_pack,
        ctx_id: int = settings.detector.ctx_id,
        det_size: tuple[int, int] = settings.detector.detection_size,
    ) -> None:
        self._model_pack = model_pack
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._app = self._load_model()

    def _load_model(self):
        """
        Lazily imports insightface (heavy import, avoid at module load time
        for faster CLI --help / test collection) and initializes the
        detection + recognition model pack.

        We ask FaceAnalysis for the full pack (det + recognition) here
        rather than a detector-only model, because embedding.py reuses
        the SAME app instance to avoid loading ArcFace weights twice.
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is not installed. Run `pip install insightface onnxruntime`."
            ) from exc

        logger.info("Loading InsightFace model pack '%s' (ctx_id=%d)...", self._model_pack, self._ctx_id)
        app = FaceAnalysis(name=self._model_pack, providers=self._providers())
        app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        logger.info("InsightFace model loaded successfully.")
        return app

    def _providers(self) -> list[str]:
        """Pick ONNX Runtime execution providers based on config.ctx_id."""
        if self._ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def app(self):
        """Expose the underlying FaceAnalysis app so embedding.py can reuse it."""
        return self._app

    def detect(self, image: np.ndarray, source_name: str = "") -> list[DetectedFace]:
        """
        Detect all faces in a single BGR image.

        Returns an empty list (never raises) if:
          - no face is found
          - the image is invalid
          - every detected face fails the min-score / min-size filters

        Multiple faces are all returned - the caller decides how to handle
        "multiple faces per image" (e.g. treat each as a separate sample).
        """
        if image is None or image.size == 0:
            logger.warning("Empty image passed to detector (%s)", source_name)
            return []

        try:
            raw_faces = self._app.get(image)
        except Exception as exc:  # noqa: BLE001 - model call boundary
            logger.error("Detection failed for %s: %s", source_name, exc)
            return []

        results: list[DetectedFace] = []
        for face in raw_faces:
            score = float(face.det_score)
            x1, y1, x2, y2 = face.bbox.astype(int)
            width, height = x2 - x1, y2 - y1

            if score < settings.detector.min_detection_score:
                logger.debug("Rejected low-confidence face (%.2f) in %s", score, source_name)
                continue
            if min(width, height) < settings.detector.min_face_pixels:
                logger.debug("Rejected too-small face (%dx%d) in %s", width, height, source_name)
                continue

            # InsightFace's FaceAnalysis already returns an aligned crop
            # via face.normed_embedding pipeline internals; we still build
            # our own aligned crop explicitly for transparency/testability.
            aligned = self._align(image, face.kps)

            results.append(
                DetectedFace(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    landmarks=np.array(face.kps, dtype=np.float32),
                    detection_score=score,
                    aligned_face=aligned,
                )
            )

        if not results:
            logger.info("No valid faces detected in %s", source_name)

        return results

    @staticmethod
    def _align(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
        """
        Warp the face into a canonical 112x112 pose using the 5-point
        landmarks, matching the alignment ArcFace was trained on.

        Uses the standard ArcFace reference landmark template and an
        affine (similarity) transform - this is why alignment matters:
        it removes in-plane rotation and scale variance BEFORE the
        embedding model ever sees the face.
        """
        import cv2

        # Canonical reference points for a 112x112 ArcFace-aligned crop.
        reference = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        transform_matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), reference)
        if transform_matrix is None:
            # Fallback: plain resize if alignment matrix estimation fails
            return cv2.resize(image, (output_size, output_size))

        aligned = cv2.warpAffine(image, transform_matrix, (output_size, output_size), borderValue=0.0)
        return aligned
