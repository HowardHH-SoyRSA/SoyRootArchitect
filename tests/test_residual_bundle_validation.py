import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soyrootbio.measurement_evidence import build_measurement_evidence


def _validator():
    path=Path(__file__).resolve().parents[1]/'scripts/validate_residual_repair.py'
    spec=importlib.util.spec_from_file_location('residual_repair_validator',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(path, *, shorter_fit=False):
    support=path/'support'; support.mkdir(parents=True)
    (path/'tables').mkdir()
    points=np.array([[0.,0.,0.],[0.,1.,0.],[0.,2.,0.],
                     [1.,0.,0.],[1.,1.,0.],[1.,2.,0.]])
    faces=np.array([[0,1,3],[1,4,3],[1,2,4],[2,5,4]])
    labels=np.array([0,0,0,1,1,1])
    np.savez(support/'analysis_geometry.npz',positions=points,triangles=faces,
             root_labels=labels,root_orders=labels,assignment_states=np.zeros(6,int),
             colors=np.zeros((6,3),np.uint8))
    np.savez(support/'ownership_evidence_masks.npz',above_base_excluded=np.zeros(6,bool))
    np.savez(support/'presentation_noise_masks.npz',excluded_full_vertices=np.zeros(6,bool))
    roots=[]; rows=[]
    for label,rid in enumerate(['primary','child']):
        short=shorter_fit and label==1
        selected=np.array([3,4]) if short else np.flatnonzero(labels==label)
        line=points[selected[[0,-1]]]
        evidence=build_measurement_evidence(points,labels,label,line,triangles=faces,
            selected_vertex_indices=selected,spacing=.5)
        roots.append({'root_id':rid,'parent_id':None if label==0 else 'primary',
            'root_order':label,'body_start_index':0,'polyline':line.tolist(),
            'centerline_assessment':{'measurement_evidence':evidence}})
        rows.append({'root_id':rid,'length':1. if short else 2.,
            'mean_diameter':.25 if short else .2,'volume':.05 if short else .1,
            'length_measurement_eligible':True,'length_measurement_status':'partial',
            'mean_diameter_measurement_status':'partial','volume_measurement_status':'partial'})
    (support/'root_hierarchy.json').write_text(json.dumps({'roots':roots}))
    (support/'metadata.json').write_text(json.dumps({'coordinate_unit':'mm',
        'primary_top_reference_mm':[0.,0.,0.],
        'root_label_map':[{'root_id':r['root_id'],'numeric_label':i} for i,r in enumerate(roots)]}))
    pd.DataFrame(rows).to_csv(path/'tables/root_inventory.csv',index=False)
    return path


def test_residual_validator_reports_selection_loss_despite_unchanged_ownership(tmp_path):
    before=_bundle(tmp_path/'before')
    after=_bundle(tmp_path/'after',shorter_fit=True)
    result=_validator().compare(before,after)
    assert result['passed']
    assert result['changed_ownership_vertices']==0
    assert result['previously_measured_lateral_support_preserved']
    assert result['roots_with_selected_support_loss']==['child']
    assert result['selected_support_changes_require_review']
    child=next(row for row in result['roots'] if row['root_id']=='child')
    assert child['measurement_support_retained_in_new_selection']==2
    assert child['not_selected_support_vertex_indices']==[5]
    assert child['measurements']['length']['delta']==-1.
    assert child['measurements']['volume']['delta']==-.05


def test_residual_validator_self_comparison_has_no_selection_or_metric_changes(tmp_path):
    bundle=_bundle(tmp_path/'bundle')
    result=_validator().compare(bundle,bundle)
    assert result['selected_support_not_retained_count']==0
    assert result['selected_support_added_count']==0
    assert not result['selected_support_changes_require_review']
    assert all(metric['delta']==0 for row in result['roots'] for metric in row['measurements'].values())
