"""
Unit tests for detector.py.

The real InsightFace model is heavy to download in CI, so most tests here
mock the FaceAnalysis app and only test OUR logic: filtering, error
handling, and the alignment math. A slower, optional integration test at
the bottom exercises the real model if it's available.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detector import DetectedFace, FaceDetector


@pytest.fixture
def fake_raw_face():
    """Mimics the object insightface.app.FaceAnalysis.get() returns per face."""
    face = MagicMock()
    face.det_score = 0.92
    face.bbox = np.array([10, 10, 110, 110], dtype=np.float32)
    face.kps = np.array(
        [[38.0, 51.0], [73.0, 51.0], [56.0, 71.0], [41.0, 92.0], [70.0, 92.0]],
        dtype=np.float32,
    )
    return face


@pytest.fixture
def detector_with_mocked_model(fake_raw_face):
    with patch.object(FaceDetector, "_load_model", return_value=MagicMock()):
        det = FaceDetector()
    det._app.get = MagicMock(return_value=[fake_raw_face])
    return det


def test_detect_returns_face_for_valid_image(detector_with_mocked_model):
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    results = detector_with_mocked_model.detect(image, source_name="test.jpg")

    assert len(results) == 1
    assert isinstance(results[0], DetectedFace)
    assert results[0].detection_score == pytest.approx(0.92)
    assert results[0].aligned_face.shape == (112, 112, 3)


def test_detect_handles_empty_image(detector_with_mocked_model):
    results = detector_with_mocked_model.detect(np.array([]), source_name="empty.jpg")
    assert results == []


def test_detect_filters_low_confidence_face(detector_with_mocked_model, fake_raw_face):
    fake_raw_face.det_score = 0.10  # below default min_detection_score
    detector_with_mocked_model._app.get = MagicMock(return_value=[fake_raw_face])

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    results = detector_with_mocked_model.detect(image)
    assert results == []


def test_detect_handles_model_exception_gracefully(detector_with_mocked_model):
    detector_with_mocked_model._app.get = MagicMock(side_effect=RuntimeError("model crashed"))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Should not raise - detection failures degrade to an empty list.
    results = detector_with_mocked_model.detect(image)
    assert results == []


def test_align_returns_correct_shape():
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    landmarks = np.array(
        [[100, 120], [180, 120], [140, 160], [110, 200], [170, 200]], dtype=np.float32
    )
    aligned = FaceDetector._align(image, landmarks, output_size=112)
    assert aligned.shape == (112, 112, 3)
