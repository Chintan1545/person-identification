import numpy as np
import pytest

from confidence import ConfidenceScorer


@pytest.fixture
def simple_two_cluster_data():
    # Cluster 0: two identical unit vectors
    v0 = np.array([1.0, 0.0, 0.0])
    # Cluster 1: two identical, orthogonal unit vectors
    v1 = np.array([0.0, 1.0, 0.0])

    embeddings = np.array([v0, v0, v1, v1])
    labels = np.array([0, 0, 1, 1])
    centroids = {0: v0, 1: v1}
    return embeddings, labels, centroids


def test_centroid_confidence_perfect_match(simple_two_cluster_data):
    embeddings, labels, centroids = simple_two_cluster_data
    scorer = ConfidenceScorer(method="centroid")
    scores = scorer.score_all(embeddings, labels, centroids)

    # Every point is identical to its cluster centroid -> similarity 1.0 -> 100 confidence
    np.testing.assert_allclose(scores, [100.0, 100.0, 100.0, 100.0], atol=1e-3)


def test_noise_points_get_fixed_low_confidence():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([-1, -1])
    scorer = ConfidenceScorer(method="centroid", noise_confidence=0.0)
    scores = scorer.score_all(embeddings, labels, centroids={})
    np.testing.assert_array_equal(scores, [0.0, 0.0])


def test_confidence_scores_are_bounded_0_to_100():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 512))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=20)

    centroids = {}
    for cid in set(labels):
        members = embeddings[labels == cid]
        c = members.mean(axis=0)
        centroids[cid] = c / np.linalg.norm(c)

    scorer = ConfidenceScorer(method="centroid")
    scores = scorer.score_all(embeddings, labels, centroids)

    assert scores.min() >= 0.0
    assert scores.max() <= 100.0


def test_knn_method_runs_without_error(simple_two_cluster_data):
    embeddings, labels, centroids = simple_two_cluster_data
    scorer = ConfidenceScorer(method="knn", knn_k=1)
    scores = scorer.score_all(embeddings, labels, centroids)
    assert len(scores) == len(embeddings)
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        ConfidenceScorer(method="not_a_real_method")
