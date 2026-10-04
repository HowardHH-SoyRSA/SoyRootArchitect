from pathlib import Path

import numpy as np

from soyrootbio.geometry import normalize_unit_box
from soyrootbio.pipeline import _full_normalized_geometry
from soyrootbio.types import PointCloudData


def test_full_normalized_geometry_reuses_exact_identity_mapping_only():
    full = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 2.0], [1.0, 0.5, 2.0], [2.0, 1.0, 3.0]]
    )
    analysis = full.copy()
    normalized, normalization = normalize_unit_box(analysis)
    cloud = PointCloudData(
        points=analysis,
        full_points=full,
        analysis_indices=np.arange(len(full)),
        source_path=Path("unchanged.ply"),
        original_points=full.copy(),
    )
    assert _full_normalized_geometry(cloud, normalized, normalization) is normalized
    np.testing.assert_array_equal(cloud.export_points, full)
    np.testing.assert_array_equal(cloud.original_points, full)
    np.testing.assert_array_equal(cloud.analysis_indices, np.arange(len(full)))

    # An equal-size analysis array can still have a different vertex order.
    cloud.points = full[[0, 2, 1, 3]].copy()
    cloud.points[1, 0] += 0.125
    reordered, reordered_norm = normalize_unit_box(cloud.points)
    transformed = _full_normalized_geometry(cloud, reordered, reordered_norm)
    assert transformed is not reordered
    np.testing.assert_array_equal(transformed, reordered_norm.transform_points(full))

    # A sampled analysis cloud must transform the independent full array.
    cloud.points = full[[0, 2, 3]].copy()
    cloud.analysis_indices = np.array([0, 2, 3])
    sampled, sampled_norm = normalize_unit_box(cloud.points)
    transformed = _full_normalized_geometry(cloud, sampled, sampled_norm)
    assert transformed is not sampled
    np.testing.assert_array_equal(transformed, sampled_norm.transform_points(full))
