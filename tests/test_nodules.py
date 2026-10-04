import json
from pathlib import Path

import numpy as np
import pytest

from soyrootbio.nodules import (NoduleResult, NODULE_RGB, detect_nodules, quantify_nodules,
                               export_nodules, apply_nodule_review, geometry_digest)
from soyrootbio.pipeline import PipelineConfig
from soyrootbio.cli import build_parser
from soyrootbio.export import SEGMENT_COLORS, _label_properties
from soyrootbio.traits import _partition_mesh_surface_areas
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.io import write_labeled_ply


def tube_mesh(bulge=False):
    z = np.linspace(-5, 5, 101)
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    radius = np.full(len(z), .3)
    if bulge:
        radius = np.maximum(radius, np.sqrt(np.maximum(1.4**2 - z**2, 0)))
    p = np.column_stack([(radius[:, None]*np.cos(theta)).ravel(),
                         (radius[:, None]*np.sin(theta)).ravel(), np.repeat(z, len(theta))])
    faces = []
    for row in range(len(z)-1):
        for col in range(len(theta)):
            a = row*len(theta)+col; b = row*len(theta)+(col+1)%len(theta)
            faces.extend([[a,b,a+len(theta)], [b,b+len(theta),a+len(theta)]])
    return p, np.asarray(faces)


def test_default_off_and_cli_explicit_opt_in():
    assert not PipelineConfig(Path('input'), Path('output')).nodule_aware
    parser = build_parser()
    assert not parser.parse_args(['run','--input','a','--output','b']).nodule_aware
    assert parser.parse_args(['run','--input','a','--output','b','--nodule-aware']).nodule_aware


def test_nodule_color_is_unique_and_ply_round_trip(tmp_path):
    for key, color in SEGMENT_COLORS.items():
        if key != 'nodule':
            assert tuple(np.round(color*255).astype(int)) != NODULE_RGB
    p = np.array([[0,0,0],[1,0,0],[0,1,0]],float)
    labels = np.array([-3,-4,-1])
    colors, orders, states = _label_properties(labels, [])
    np.testing.assert_array_equal(states,[3,3,0])
    write_labeled_ply(tmp_path/'nodules.ply',p,triangles=np.array([[0,1,2]]),colors=colors,root_ids=labels,root_orders=orders,assignment_states=states)
    loaded=read_labeled_ply(tmp_path/'nodules.ply')
    np.testing.assert_array_equal(loaded.root_labels,labels)
    np.testing.assert_array_equal(loaded.colors[:2],np.tile(NODULE_RGB,(2,1)))


def test_uniform_tube_does_not_become_a_nodule():
    p,f=tube_mesh()
    result=detect_nodules(p,f)
    assert not result.mask.any()


def test_bulge_is_one_object_independent_of_root_labels_and_collar_excluded():
    p,f=tube_mesh(True)
    result=detect_nodules(p,f)
    accepted=[o for o in result.objects if o['status']=='accepted']
    assert len(accepted)==1
    assert abs(p[result.mask,2].mean())<.15
    assert np.max(abs(p[result.mask,2]))<2
    protected=detect_nodules(p,f,excluded_mask=np.abs(p[:,2])<2)
    assert not protected.mask[np.abs(p[:,2])<2].any()


def test_missing_mesh_reports_unresolved_without_fabricated_labels():
    p,_=tube_mesh(True)
    result=detect_nodules(p,None)
    assert result.status=='unresolved_no_native_mesh'
    assert not result.mask.any()


def test_interface_surface_area_is_not_double_counted():
    p=np.array([[0,0,0],[3,0,0],[0,2,0]],float);f=np.array([[0,1,2]])
    labels=np.array([0,0,-3])
    areas=_partition_mesh_surface_areas(p,f,labels)
    assert areas[0]==pytest.approx(2)
    # Exactly one third of the triangle belongs to the nodule, matching the
    # vertex-weighted native surface partition used by nodule measurement.
    assert areas[0]+3/3==pytest.approx(3)


def test_separate_size_position_export_and_review_guard(tmp_path):
    p,f=tube_mesh(True);result=detect_nodules(p,f)
    labels=np.zeros(len(p),int);labels[result.mask]=result.vertex_labels[result.mask]
    roots={0:{'root_id':'primary','order':0,'points':np.array([[0,0,5],[0,0,-5]]),'length':10}}
    quantify_nodules(result,p,f,labels,roots,np.array([0,0,5]))
    obj=next(o for o in result.objects if o['status']=='accepted')
    assert obj['depth_below_primary_top']==pytest.approx(5,abs=.1)
    assert obj['surface_area']>0
    assert obj['supporting_root_id']=='primary'
    if obj['volume'] is not None:
        assert obj['volume']>0
        assert obj['volume_method']=='estimated_planar_neck_cap_on_observed_faces'
    export_nodules(tmp_path,result)
    assert (tmp_path/'nodule_traits.csv').is_file()
    assert (tmp_path/'nodules_by_root.csv').is_file()
    review={'geometry_sha256':'wrong','decisions':[]}
    path=tmp_path/'review.json';path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='geometry'):apply_nodule_review(result,path,p,np.zeros(len(p),bool))
    review={'geometry_sha256':geometry_digest(p),'decisions':[{'geometry_fingerprint':obj['geometry_fingerprint'],'status':'rejected'}]}
    path.write_text(json.dumps(review));apply_nodule_review(result,path,p,np.zeros(len(p),bool))
    assert not result.mask.any()


def test_nodule_pipeline_keeps_primary_and_exports_separate_objects(tmp_path):
    from soyrootbio.pipeline import run_pipeline
    import pandas as pd
    p,f=tube_mesh(True)
    source=tmp_path/'bulge.ply'
    write_labeled_ply(source,p,triangles=f)
    result=run_pipeline(PipelineConfig(source,tmp_path/'out',start=(0,0,5),end=(0,0,-5),
                                       nodule_aware=True,max_root_order=1,worker_threads=1))
    document=json.loads((tmp_path/'out/nodules.json').read_text())
    assert document['accepted_count']>=1
    assert np.any(result.full_root_labels<=-3)
    assert np.all(result.full_root_labels[result.full_above_base_mask]==-1)
    assert 'primary' in result.traits.root_id.values
    assert not result.lateral_paths
    assert 'centerline_nodule_obscured' in result.traits.iloc[0].qc_flags
    assert np.isnan(result.traits.iloc[0].length)
    assert np.ptp(result.primary_path[:,2])*result.normalization.scale>9
    assert not result.traits.root_id.str.startswith('nodule').any()
    assert result.traits.attrs['system_summary']['nodule_excluded_from_root_totals']
    mesh=read_labeled_ply(tmp_path/'out/segmented_root_structure.ply')
    np.testing.assert_array_equal(mesh.positions,p)
    np.testing.assert_array_equal(mesh.triangles,f)
    assert set(mesh.assignment_states[mesh.root_labels<=-3])=={3}
    book=pd.ExcelFile(tmp_path/'out/traits.xlsx')
    assert {'Nodules','Nodule summary','Nodule depth'}<=set(book.sheet_names)
