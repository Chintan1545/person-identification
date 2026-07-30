import numpy as np
import pytest

from clustering import FaceClusterer


def _make_person_cluster(base_vector: np.ndarray, n: int, noise_scale: float = 0.02) -> np.ndarray:
    """Generate n embeddings that are small random perturbations of a base vector, then L2-normalized."""
    vectors = base_vector + np.random.normal(scale=noise_scale, size=(n, len(base_vector)))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def two_person_embeddings():
    rng = np.random.default_rng(42)
    person_a_base = rng.normal(size=512)
    person_a_base /= np.linalg.norm(person_a_base)

    # Person B: force a very different, roughly orthogonal direction
    person_b_base = rng.normal(size=512)
    person_b_base -= person_b_base.dot(person_a_base) * person_a_base
    person_b_base /= np.linalg.norm(person_b_base)

    a = _make_person_cluster(person_a_base, 5)
    b = _make_person_cluster(person_b_base, 5)
    embeddings = np.vstack([a, b])
    return embeddings


def test_fit_separates_two_distinct_people(two_person_embeddings):
    clusterer = FaceClusterer(eps=0.4, min_samples=2)
    result = clusterer.fit(two_person_embeddings)

    assert result.n_clusters == 2
    # first 5 rows should share one label, last 5 rows another, and they differ
    assert len(set(result.labels[:5])) == 1
    assert len(set(result.labels[5:])) == 1
    assert result.labels[0] != result.labels[5]


def test_fit_handles_empty_input():
    clusterer = FaceClusterer()
    result = clusterer.fit(np.array([]).reshape(0, 512))
    assert result.n_clusters == 0
    assert result.n_noise == 0
    assert len(result.labels) == 0


def test_fit_marks_outlier_as_noise(two_person_embeddings):
    rng = np.random.default_rng(7)
    outlier = rng.normal(size=(1, 512))
    outlier /= np.linalg.norm(outlier)

    embeddings = np.vstack([two_person_embeddings, outlier])
    clusterer = FaceClusterer(eps=0.3, min_samples=2)
    result = clusterer.fit(embeddings)

    assert result.labels[-1] == -1  # the single dissimilar point should be noise


def test_fit_raises_on_wrong_dimensionality():
    clusterer = FaceClusterer()
    with pytest.raises(ValueError):
        clusterer.fit(np.zeros(10))  # 1D instead of 2D


def test_cluster_centroids_are_unit_normalized(two_person_embeddings):
    clusterer = FaceClusterer(eps=0.4, min_samples=2)
    result = clusterer.fit(two_person_embeddings)
    centroids = clusterer.cluster_centroids(two_person_embeddings, result.labels)

    assert len(centroids) == result.n_clusters
    for centroid in centroids.values():
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-4)
