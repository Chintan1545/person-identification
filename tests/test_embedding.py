from unittest.mock import MagicMock

import numpy as np
import pytest

from embedding import EmbeddingExtractor, cosine_distance, cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_distance_is_one_minus_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(0.0)


def test_l2_normalize_produces_unit_vector():
    v = np.array([3.0, 4.0])  # norm = 5
    normalized = EmbeddingExtractor._l2_normalize(v)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    np.testing.assert_allclose(normalized, [0.6, 0.8])


def test_l2_normalize_handles_zero_vector():
    v = np.zeros(512)
    normalized = EmbeddingExtractor._l2_normalize(v)
    np.testing.assert_array_equal(normalized, v)  # returned as-is, no NaNs


def test_extract_uses_cache(tmp_path):
    fake_detector = MagicMock()
    fake_detector.app.models = {"recognition": MagicMock(get_feat=MagicMock(
        return_value=np.random.rand(1, 512).astype(np.float32)
    ))}

    extractor = EmbeddingExtractor(detector=fake_detector, use_cache=True)
    extractor._cache_dir = tmp_path  # redirect cache to a temp dir

    face = np.zeros((112, 112, 3), dtype=np.uint8)
    first = extractor.extract(face, cache_key="unit_test_key")
    second = extractor.extract(face, cache_key="unit_test_key")

    # get_feat should only be called once - second call hits the cache
    assert fake_detector.app.models["recognition"].get_feat.call_count == 1
    np.testing.assert_array_equal(first, second)
