from pathlib import Path

import numpy as np
import pytest

from soyrootbio.io import _prepare_geometry, _mesh_audit, require_open3d
from soyrootbio.stl_geometry import index_stl_facets
from test_io import _unit_cube


def faceted(points, faces):
    records = points[faces].reshape(-1, 3)
    return records, np.arange(len(records)).reshape(-1, 3)


def test_closed_stl_exact_indexing_preserves_oriented_faces_and_original_mapping():
    cube, faces = _unit_cube()
    points = np.vstack([cube, cube + [3, 0, 0]])
    faces = np.vstack([faces, faces + 8])
    source, facets = faceted(points, faces)
    cloud = _prepare_geometry(source, facets, path=Path('closed.stl'), sample_points=0,
                              random_seed=42, runtime_limit_seconds=1800,
                              minimum_retained_fraction=.25, input_mode='auto', progress_callback=None)
    assert len(cloud.points) == len(cloud.full_points) == 16
    assert cloud.source_metadata['stl_indexing']['status'] == 'supported_exact_seams'
    np.testing.assert_array_equal(cloud.original_points, source)
    np.testing.assert_array_equal(cloud.original_triangles, facets)
    mapping = cloud.geometry_mapping
    np.testing.assert_array_equal(cloud.full_points[mapping['original_to_full']], source)
    np.testing.assert_array_equal(source[mapping['full_to_original']], cloud.full_points)
    np.testing.assert_array_equal(cloud.full_points[cloud.triangles], source[facets])
    np.testing.assert_array_equal(mapping['original_face_to_full'], np.arange(len(faces)))
    audit = _mesh_audit(cloud.full_points, cloud.triangles)
    assert audit['connected_component_count'] == 2
    assert audit['volume_reliable']
    assert audit['absolute_volume_source_units3'] == pytest.approx(2)


def test_touching_closed_shells_do_not_acquire_a_vertex_bridge():
    cube, faces = _unit_cube()
    points = np.vstack([cube, cube + [1, 1, 1]])
    faces = np.vstack([faces, faces + 8])
    source, facets = faceted(points, faces)
    full, triangles, pool, forward, reverse, unresolved, report = index_stl_facets(source, facets, require_open3d())
    assert report['status'] == 'unresolved_stl_connectivity'
    assert report['unresolved_position_count'] == 1
    assert len(full) > 16  # No stitching across the ambiguous contact.
    assert len(pool) == 15  # Only one density observation at the shared position.
    assert not set(triangles[:12].ravel()) & set(triangles[12:].ravel())
    np.testing.assert_array_equal(full[triangles], source[facets])
    assert len(unresolved) > 1


@pytest.mark.parametrize('case', ['open', 'overlap', 'orientation'])
def test_unsupported_seams_stay_unresolved(case):
    points, faces = _unit_cube()
    if case == 'open':
        faces = faces[2:]
    elif case == 'overlap':
        faces = np.vstack([faces, faces[:, ::-1]])
    else:
        faces[0] = faces[0, ::-1]
    source, facets = faceted(points, faces)
    full, triangles, pool, forward, reverse, unresolved, report = index_stl_facets(source, facets, require_open3d())
    assert report['status'] == 'unresolved_stl_connectivity'
    assert len(unresolved)
    np.testing.assert_array_equal(full[triangles], source[facets])
    assert len(pool) == len(np.unique(source, axis=0))


def test_nearby_shells_are_never_tolerance_welded():
    cube, faces = _unit_cube()
    points = np.vstack([cube, cube + [1 + 1e-9, 1, 1]])
    faces = np.vstack([faces, faces + 8])
    source, facets = faceted(points, faces)
    full, triangles, *_rest, report = index_stl_facets(source, facets, require_open3d())
    assert len(full) == 16
    assert report['status'] == 'supported_exact_seams'
    assert not set(triangles[:12].ravel()) & set(triangles[12:].ravel())


def test_stl_nonfinite_exclusion_and_cap_have_explicit_source_mappings():
    cube, faces = _unit_cube()
    source, facets = faceted(np.vstack([cube, cube + [3, 0, 0]]), np.vstack([faces, faces + 8]))
    source[0] = np.nan
    cloud = _prepare_geometry(source, facets, path=Path('invalid.stl'), sample_points=10,
                              random_seed=42, runtime_limit_seconds=1800,
                              minimum_retained_fraction=.25, input_mode='auto', progress_callback=None)
    mapping = cloud.geometry_mapping
    assert mapping['original_to_full'][0] == -1
    assert mapping['original_face_to_full'][0] == -1
    assert len(cloud.points) == 10
    np.testing.assert_array_equal(cloud.original_points, source)
    kept = mapping['original_face_to_full'] >= 0
    np.testing.assert_array_equal(cloud.full_points[cloud.triangles], source[facets[kept]])
    np.testing.assert_array_equal(cloud.full_points[mapping['analysis_to_full']], cloud.points)
