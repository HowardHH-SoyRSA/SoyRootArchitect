import numpy as np
import pandas as pd
import pytest

from soyrootbio.architecture_traits import add_architecture_traits, mesh_measurement
from soyrootbio.types import RootPath


def tetra(offset=(0, 0, 0)):
    xyz = np.array([[0.,0,0], [2,0,0], [0,3,0], [0,0,4]]) + offset
    faces = np.array([[0,2,1], [0,1,3], [0,3,2], [1,2,3]])
    return xyz, faces


def test_mesh_volume_analytic_closed_and_open_validity():
    points, faces = tetra()
    a = mesh_measurement(points, faces)
    b = mesh_measurement(points + [1000, -50, 37], faces)
    assert a['mesh_volume'] == pytest.approx(4)
    assert a['mesh_surface_area'] == pytest.approx(13 + np.sqrt(244)/2)
    assert a['mesh_volume_reliable'] and b['mesh_volume_reliable']
    assert b['mesh_volume'] == pytest.approx(a['mesh_volume'])
    assert a['mesh_volume_origin_sensitivity'] < 1e-12
    opened = mesh_measurement(points, faces[:-1])
    assert not opened['mesh_volume_reliable']
    assert opened['mesh_boundary_edges'] == 3
    assert opened['mesh_volume_origin_sensitivity'] > 0


def fixture():
    primary = np.array([[0, 0, 10.], [0, 0, 0.]])
    paths = [RootPath('a', np.array([[0,0,8.], [8,0,8.]]), parent_id='primary', order=1, insertion_point=np.array([0,0,8.])),
             RootPath('b', np.array([[0,0,5.], [6,0,5.]]), parent_id='primary', order=1, insertion_point=np.array([0,0,5.])),
             RootPath('c', np.array([[2,0,8.], [2,3,8.]]), parent_id='a', order=2, insertion_point=np.array([2,0,8.])),
             RootPath('d', np.array([[6,0,8.], [6,2,8.]]), parent_id='a', order=2, insertion_point=np.array([6,0,8.])),
             RootPath('missing', np.array([[0,0,4.], [100,0,4.]])),
             RootPath('zero', np.array([[0,0,3.], [200,0,3.]]))]
    rows = []
    for i, (rid, order, length) in enumerate([('primary',0,10), ('a',1,8), ('b',1,6), ('c',2,3), ('d',2,2), ('missing',1,np.nan), ('zero',1,0)]):
        rows.append(dict(root_id=rid, root_order=order, length=length, volume=1., mean_diameter=i+1.,
                         surface_area=1., volume_estimate_complete=True))
    xyz, faces = tetra()
    points = np.vstack([xyz, xyz+[0,0,5], xyz+[1000,0,0], xyz+[-1000,0,0]])
    labels = np.repeat([0, 3, 5, -1],4)
    faces = np.vstack([faces+i*4 for i in range(4)])
    return pd.DataFrame(rows), primary, paths, points, labels, faces


def test_all_orders_filter_segmented_geometry_and_parent_spacing():
    frame, primary, paths, points, labels, faces = fixture()
    original = points.copy(), labels.copy()
    add_architecture_traits(frame, primary, paths, points, labels, faces, np.array([0,0,15.]), np.array([0,0,-1.]), 'mm')
    summary = {r['scope']: r for r in frame.attrs['architecture_summary']}
    system = summary['system']
    assert system['root_count'] == 5 and system['excluded_root_count'] == 2
    assert system['length_sum'] == 29
    assert system['mesh_volume'] == pytest.approx(8)
    assert system['max_width'] < 4
    assert system['max_depth'] == 15  # immutable top, not shortened centerline start (10)
    lateral = summary['laterals']
    assert lateral['lateral_density'] == pytest.approx(4/18)  # distinct parents: primary + a
    assert lateral['mean_interbranch'] == pytest.approx(3.5) # gaps 3 and 4; never cross parents
    assert summary['order_2']['mean_interbranch'] == 4
    assert summary['order_2']['diameter_median'] == 4.5
    assert summary['order_2']['diameter_p10'] == pytest.approx(4.1)
    np.testing.assert_array_equal(points, original[0]); np.testing.assert_array_equal(labels, original[1])


def test_coincident_positions_and_invalid_parent_are_explicit():
    frame, primary, paths, points, labels, faces = fixture()
    paths[1].insertion_point = paths[0].insertion_point.copy()
    frame.loc[frame.root_id == 'a', 'length'] = np.nan
    add_architecture_traits(frame, primary, paths, points, labels, faces, primary[0], np.array([0,0,-1.]), 'mm')
    result = {r['scope']:r for r in frame.attrs['architecture_summary']}['laterals']
    assert result['branching_unusable_parent_count'] == 2
    assert np.isnan(result['lateral_density'])
    frame, primary, paths, points, labels, faces = fixture()
    paths[1].insertion_point = paths[0].insertion_point.copy()
    add_architecture_traits(frame, primary, paths, points, labels, faces, primary[0], np.array([0,0,-1.]), 'mm')
    result = {r['scope']:r for r in frame.attrs['architecture_summary']}['laterals']
    assert result['mean_interbranch'] == 2
    assert result['coincident_gap_count'] == 1
    assert result['distinct_site_mean_interbranch'] == 4


def test_segmented_union_accepts_cross_root_faces_but_excludes_nonroot_corners():
    frame, primary, paths, points, labels, faces = fixture()
    # One tetra spans primary and an eligible O2 root. Its union is closed;
    # neither individual root owns a closed tetra. The distant unassigned
    # tetra and the missing-length root cannot contribute.
    labels[:4] = [0, 3, 0, 3]
    labels[4:8] = -3  # nodule surface is not root volume
    add_architecture_traits(frame, primary, paths, points, labels, faces, primary[0], np.array([0,0,-1.]), 'mm')
    scopes = {r['scope']:r for r in frame.attrs['architecture_summary']}
    assert scopes['system']['mesh_volume'] == pytest.approx(4.)
    assert scopes['system']['mesh_volume_reliable']
    assert scopes['system']['mesh_triangle_count'] == 4
    assert np.isnan(scopes['order_0']['mesh_volume'])
    assert np.isnan(scopes['order_2']['mesh_volume'])
