import numpy as np
import pytest

from soyrootbio.io import _mesh_audit
from soyrootbio.traits import compute_traits, lateral_counts_frame
from soyrootbio.types import Normalization, RootPath


def test_noise_geometry_does_not_contribute_to_measurements_or_fraction_denominator():
    main = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    points = np.vstack((main, main*20+100))
    triangles = np.vstack((faces, faces+4))
    labels = np.array([0]*4+[-1]*4)
    mask = labels == -1
    primary = main[[0,3]]
    def measure(p, f, lab, noise):
        return compute_traits(primary, [], p, lab==0, lab, Normalization(np.zeros(3),1),
            full_points=p, triangles=f, full_root_labels=lab,
            mesh_metadata=_mesh_audit(p,f), noise_mask=noise)
    actual = measure(points,triangles,labels,mask)
    expected = measure(main,faces,labels[:4],None)
    for key in ('length','surface_area','volume','mean_radius','point_count'):
        np.testing.assert_allclose(actual[key],expected[key])
    for key in ('root_system_surface_area','root_system_volume','assigned_vertex_fraction','unassigned_vertex_fraction'):
        assert actual.attrs['system_summary'][key] == expected.attrs['system_summary'][key]
    labels[4]=0
    with pytest.raises(ValueError,match='cannot contribute'):
        measure(points,triangles,labels,mask)


def test_noise_only_topology_parent_is_excluded_from_traits_and_root_counts():
    points = np.array([[0.,0,0],[0,0,1],[0,0,2]])
    label = np.array([0,0,0])
    root = RootPath('noise-parent', points=np.array([[0.,0,1],[50,0,1]]),
                    qc_flags=['noise_excluded_root'])
    traits = compute_traits(points,[root],points,label==0,label,Normalization(np.zeros(3),1),
                            full_points=points,full_root_labels=label)
    row = traits.iloc[1]
    assert row.measurement_excluded and row.point_count == 0
    assert np.isnan(row.length) and np.isnan(row.surface_area) and np.isnan(row.volume)
    assert np.isnan(row.tip_angle_primary_deg)
    assert traits.attrs['system_summary']['root_count_total'] == 1
    assert traits.iloc[0].selected_lateral_count == 0
    assert lateral_counts_frame(traits).empty
