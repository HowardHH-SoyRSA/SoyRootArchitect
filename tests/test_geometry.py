import numpy as np

from soyrootbio.geometry import mean_nearest_neighbor_distance, normalize_unit_box, path_length, resample_polyline


def test_normalize_unit_box_and_inverse():
    points = np.array([[2.0, 4.0, 10.0], [4.0, 8.0, 14.0], [6.0, 6.0, 18.0]])
    normalized, transform = normalize_unit_box(points)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(np.ptp(normalized, axis=0).max(), 1.0)
    np.testing.assert_allclose(transform.inverse_points(normalized), points)


def test_mean_nearest_neighbor_distance_line():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.isclose(mean_nearest_neighbor_distance(points), 1.0)


def test_resample_polyline_preserves_length_endpoints():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    resampled = resample_polyline(points, spacing=0.25)
    np.testing.assert_allclose(resampled[0], points[0])
    np.testing.assert_allclose(resampled[-1], points[-1])
    assert np.isclose(path_length(resampled), 1.0)
