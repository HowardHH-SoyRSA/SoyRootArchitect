from pathlib import Path
import json

import numpy as np
import pytest

from soyrootbio import noise
from soyrootbio.io import load_root_geometry, write_labeled_ply
from soyrootbio.noise import detect_disconnected_noise, filter_preloaded_cloud
from soyrootbio.pipeline import PipelineConfig, run_pipeline
from soyrootbio.types import PointCloudData


def _tetra(scale, offset):
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale + offset
    return points, np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])


def _join(*meshes):
    offsets = np.cumsum([0] + [len(p) for p, f in meshes])
    return np.concatenate([p for p, f in meshes]), np.concatenate([f + offset for (p, f), offset in zip(meshes, offsets)])


def test_native_filter_preserves_connected_body_open_large_and_elongated_fragments():
    main = _tetra(100., np.zeros(3))
    speck = _tetra(.1, [101., 0., 0.])
    large = _tetra(3., [110., 0., 0.])
    thin_points, thin_faces = _tetra(1., [115., 0., 0.])
    thin_points[:, 1:] *= .001
    open_points, open_faces = _tetra(.1, [120., 0., 0.])
    p, f = _join(main, speck, large, (thin_points, thin_faces), (open_points, open_faces[:-1]))
    mask, report = detect_disconnected_noise(p, f)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(4, 8))
    assert report['excluded_component_count'] == 1
    assert report['retained_review_component_count'] == 3
    # A single native contact connects the speck; no graph-distance threshold
    # is allowed to cut this edge and call it disconnected noise.
    connected_faces = np.vstack([f, [0, 4, 5]])
    mask, _ = detect_disconnected_noise(p, connected_faces)
    assert not mask.any()


def test_native_filter_retains_split_seams_nonmanifold_and_non_dominant_meshes():
    main = _tetra(100., np.zeros(3))
    coincident = _tetra(.1, np.zeros(3))
    ambiguous = _tetra(.1, [105., 0., 0.])
    p, f = _join(main, coincident, (ambiguous[0], np.vstack([ambiguous[1], ambiguous[1][0]])))
    mask, _ = detect_disconnected_noise(p, f)
    assert not mask.any()
    p, f = _join(main, _tetra(100., [200., 0, 0]), _tetra(.1, [300., 0, 0]))
    mask, report = detect_disconnected_noise(p, f)
    assert not mask.any()
    assert {r['reason'] for r in report['components']} == {'retained_no_dominant_structure'}


def test_native_filter_scale_translation_and_vertex_order_invariant():
    p, f = _join(_tetra(100., np.zeros(3)), _tetra(.1, [101., 0, 0]))
    expected, _ = detect_disconnected_noise(p, f)
    order = np.array([7, 1, 5, 0, 2, 4, 3, 6])
    inverse = np.argsort(order)
    for scale in (.001, 1000.):
        actual, _ = detect_disconnected_noise(p[order] * scale + [12, -30, 50], inverse[f[::-1]])
        np.testing.assert_array_equal(actual, expected[order])


def test_native_filter_preserves_internal_shells_and_intersecting_external_candidates():
    # All four tetrahedra are separate in the native graph. Only one is an
    # exterior fragment; a graph-only classifier would also delete the cavity.
    p, f = _join(_tetra(100., np.zeros(3)), _tetra(.1, [10.,10,10]),
                 _tetra(.1, [101.,0,0]), _tetra(.1, [99.99,0,0]))
    mask, report = detect_disconnected_noise(p, f)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(8,12))
    reasons = {r['first_vertex']:r['reason'] for r in report['components']}
    assert reasons[4] == 'retained_internal_surface_requires_review'
    assert reasons[12] == 'retained_geometric_contact_requires_review'


def _tube_with_noise():
    n = 16
    z = np.linspace(0, 100, 41)
    theta = np.arange(n)*2*np.pi/n
    p = np.column_stack([np.tile(2*np.cos(theta),len(z)), np.tile(2*np.sin(theta),len(z)),np.repeat(z,n)])
    p = np.vstack([p, [0.,0,0], [0.,0,100]])
    f = []
    for row in range(len(z)-1):
        for col in range(n):
            a,b = row*n+col,row*n+(col+1)%n
            f.extend([[a,b,a+n],[b,b+n,a+n]])
    for col in range(n):
        f.extend([[len(p)-2,(col+1)%n,col], [len(p)-1,(len(z)-1)*n+col,(len(z)-1)*n+(col+1)%n]])
    return _join((p,np.array(f)),_tetra(.1,[0.,0,110.]))


def test_loading_filters_before_cap_preserves_all_source_geometry_and_off_bypasses(tmp_path, monkeypatch):
    p, f = _tube_with_noise()
    source = tmp_path/'input.ply'
    write_labeled_ply(source,p,triangles=f)
    cloud = load_root_geometry(source,sample_points=100,noise_reduction=True)
    assert len(cloud.points) == 100
    assert cloud.noise_mask[-4:].all()
    assert not cloud.noise_mask[cloud.analysis_indices].any()
    np.testing.assert_array_equal(cloud.export_points,p)
    np.testing.assert_array_equal(cloud.original_points,p)
    np.testing.assert_array_equal(cloud.triangles,f)
    np.testing.assert_array_equal(cloud.original_triangles,f)
    np.testing.assert_array_equal(cloud.points,p[cloud.analysis_indices])
    # OFF retains exactly the existing low-level load path and must not detect.
    monkeypatch.setattr(noise,'detect_disconnected_noise',lambda *a,**k: pytest.fail('off invoked detector'))
    off = load_root_geometry(source,sample_points=len(p),noise_reduction=False)
    np.testing.assert_array_equal(off.points,p)
    assert off.noise_mask is None


def test_preloaded_cloud_is_not_mutated_and_point_only_inputs_remain_unresolved():
    p, f = _tube_with_noise()
    cloud = PointCloudData(points=p,source_path=Path('input.ply'),triangles=f)
    filtered = filter_preloaded_cloud(cloud,enabled=True)
    assert filtered is not cloud
    assert len(cloud.points) == len(p)
    assert cloud.noise_mask is None
    assert len(filtered.points) == len(p)-4
    np.testing.assert_array_equal(filtered.export_points,p)
    assert filter_preloaded_cloud(filtered,enabled=True) is filtered
    with pytest.raises(ValueError,match='Reload'):
        filter_preloaded_cloud(filtered,enabled=False)
    mask, report = detect_disconnected_noise(p,None)
    assert not mask.any()
    assert report['status'] == 'unresolved_no_native_connectivity'


def test_pipeline_noise_stays_excluded_and_mesh_totals_exclude_its_faces(tmp_path, monkeypatch):
    from soyrootbio import pipeline
    from soyrootbio.editor.ply import read_labeled_ply
    p, f = _tube_with_noise()
    source, output = tmp_path/'input.ply',tmp_path/'result'
    write_labeled_ply(source,p,triangles=f)
    resolve = pipeline._resolve_primary_path
    def check_primary_points(points, *args, **kwargs):
        assert len(points) == len(p)-4
        assert points[:,2].max() == 100
        return resolve(points,*args,**kwargs)
    monkeypatch.setattr(pipeline,'_resolve_primary_path',check_primary_points)
    result = run_pipeline(PipelineConfig(source,output,start=(0,0,100),end=(0,0,0),
                                         max_root_order=1,worker_threads=1,sample_points=len(p)))
    assert (result.full_root_labels[-4:] == -1).all()
    assert (result.full_root_labels[result.full_above_base_mask] == -1).all()
    mesh = read_labeled_ply(output/'segmented_root_structure.ply')
    np.testing.assert_array_equal(mesh.positions,p)
    np.testing.assert_array_equal(mesh.triangles,f)
    metadata=json.loads((output/'metadata.json').read_text())
    summary=metadata['system_summary']
    expected_area=.5*np.linalg.norm(np.cross(p[f[:-4,1]]-p[f[:-4,0]],p[f[:-4,2]]-p[f[:-4,0]]),axis=1).sum()
    assert summary['root_system_surface_area'] == pytest.approx(expected_area)
    assert summary['root_system_surface_area_method'] == 'noise_filtered_mesh_triangle_area'
    assert metadata['source_geometry']['surface_area_source_units2'] > expected_area
    assert metadata['noise_reduction']['excluded_vertex_count'] == 4
    presentation = read_labeled_ply(output/'presentation_root_structure.ply')
    assert len(presentation.positions) == len(p)-4
    assert len(presentation.triangles) == len(f)-4
    with np.load(output/'noise_reduction_masks.npz') as saved:
        assert saved['excluded_full_vertices'][-4:].all()
        np.testing.assert_array_equal(p[saved['analysis_to_full']],p[:-4])


def test_cli_and_gui_switch_defaults_and_off(tmp_path):
    from soyrootbio.cli import build_parser
    from soyrootbio.desktop_gui import validate_launcher_settings
    from test_batch_guidance_gui import _app, _add_sample
    args = ['run','--input','input.ply','--output','out']
    assert build_parser().parse_args(args).noise_reduction
    assert not build_parser().parse_args(args+['--no-noise-reduction']).noise_reduction
    app = _app(tmp_path)
    entry = _add_sample(app,tmp_path)
    assert app._pipeline_config(entry,1).noise_reduction
    app.noise_reduction_var.set(False)
    assert not app._pipeline_config(entry,1).noise_reduction
    assert validate_launcher_settings(entry.input_path,tmp_path/'out',100,100).noise_reduction
    assert not validate_launcher_settings(entry.input_path,tmp_path/'out',100,100,noise_reduction=False).noise_reduction
