from __future__ import annotations
from soyrootbio.bundle_layout import bundle_file, read_bundle_mesh

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import struct
import shutil
import tkinter as tk
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from soyrootbio import hardware
from soyrootbio.batch_gui import BioInsAlgoBatchApp, PRIMARY_METHODS
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.editor.session import EditorSession, EditorValidationError
from soyrootbio.geometry import normalize_unit_box
from soyrootbio.pipeline import PipelineConfig, run_pipeline, _assign_lateral_points, _assign_full_root_labels
from soyrootbio.primary_guidance import PrimaryGuidance, read_primary_guidance, write_primary_guidance
from soyrootbio.topology import apply_hierarchy_corrections, write_editable_hierarchy
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, PointCloudData, RootPath
from soyrootbio.unit_dialog import UnitScaleDialog, UnitResolutionCancelled, resolve_batch_units
from soyrootbio.units import UnitCalibration, UnitResolutionError, resolve_unit_calibration, source_unit_metadata, validate_scale


def _stl(path, declaration):
    header = declaration.encode('utf-8').ljust(80, b'\0')
    assert len(header) == 80
    path.write_bytes(header + struct.pack('<I', 1) + b'\0' * 50)
    return path


@pytest.mark.parametrize('declaration,unit,scale', [
    ('[Unit:Millimeter] (Written by VGS 7.4.0)', 'mm', 1),
    ('[Unit: centimetres]', 'cm', 10), ('[Units:METERS]', 'm', 1000),
    ('[Unit:micrometer]', 'um', .001), ('[Unit:µm]', 'um', .001),
    ('[Unit:nm]', 'nm', .000001), ('[Unit:Inches]', 'inch', 25.4),
    ('[Unit:mm] [Unit:millimetre]', 'mm', 1),
])
def test_binary_stl_explicit_declarations(tmp_path, declaration, unit, scale):
    path = _stl(tmp_path / 'root.stl', declaration)
    calibration = resolve_unit_calibration(path)
    assert calibration.mm_per_mesh_unit == scale
    assert calibration.source == 'stl_header'
    assert calibration.declared_unit == unit
    assert calibration.raw_declaration
    assert calibration.to_dict()['output_coordinate_unit'] == 'mm'


@pytest.mark.parametrize('declaration,status', [
    ('Exported in millimeters', 'missing'), ('[Unit:voxel]', 'unsupported'),
    ('[Unit:mm][Unit:cm]', 'conflicting'), ('[Unit:mm][Unit:voxel]', 'unsupported'),
])
def test_unknown_and_conflicting_declarations_require_explicit_scale(tmp_path, declaration, status):
    path = _stl(tmp_path / 'mm_named_root.stl', declaration)
    assert source_unit_metadata(path)['unit_declaration_status'] == status
    with pytest.raises(UnitResolutionError, match='mm-per-mesh-unit'):
        resolve_unit_calibration(path)
    calibration = resolve_unit_calibration(path, .025, source='cli')
    assert calibration.mm_per_mesh_unit == .025
    assert calibration.source == 'cli'


def test_ascii_stl_and_wrong_binary_size_do_not_establish_units(tmp_path):
    path = tmp_path / 'root.stl'
    path.write_text('solid [Unit:mm]\n' + ' ' * 100 + '\nendsolid', encoding='ascii')
    assert source_unit_metadata(path)['declared_coordinate_unit'] is None
    _stl(path, '[Unit:mm]')
    path.write_bytes(path.read_bytes() + b'extra')
    assert source_unit_metadata(path)['declared_coordinate_unit'] is None


@pytest.mark.parametrize('value', [None, True, False, 0, -1, float('inf'), float('nan'), 'bad'])
def test_invalid_scales_fail_closed(value):
    with pytest.raises(UnitResolutionError):
        validate_scale(value)


def test_matching_explicit_header_scale_preserves_header_provenance_and_conflict_raises(tmp_path):
    path = _stl(tmp_path / 'root.stl', '[Unit:cm]')
    assert resolve_unit_calibration(path, 10, source='user').source == 'stl_header'
    with pytest.raises(UnitResolutionError, match='conflicts'):
        resolve_unit_calibration(path, 1)


def test_batch_rule_only_applies_to_remaining_unresolved_files_and_cancel_is_atomic(tmp_path):
    unknown = _stl(tmp_path / 'unknown.stl', 'scanner export')
    mm = _stl(tmp_path / 'mm.stl', '[Unit:mm]')
    cm = _stl(tmp_path / 'cm.stl', '[Unit:cm]')
    unknown2 = _stl(tmp_path / 'unknown2.stl', '[Unit:voxel]')
    prompts = []
    def prompt(path, **kwargs):
        prompts.append(path)
        return (.025, True)
    result = resolve_batch_units([unknown, mm, cm, unknown2], prompt)
    assert prompts == [unknown]
    assert result[unknown]['source'] == 'user'
    assert result[unknown2]['source'] == 'batch_rule'
    assert result[unknown2]['mm_per_mesh_unit'] == .025
    assert result[mm]['mm_per_mesh_unit'] == 1
    assert result[cm]['mm_per_mesh_unit'] == 10
    assert result[mm]['source'] == result[cm]['source'] == 'stl_header'
    existing = {unknown: result[unknown]}
    before = deepcopy(existing)
    with pytest.raises(UnitResolutionCancelled):
        resolve_batch_units([unknown, unknown2], lambda *_a, **_k: None, existing=existing)
    assert existing == before


def test_source_and_mm_transforms_are_explicit_and_raw_array_is_preserved():
    raw = np.array([[12., -5, 3], [14, 1, 6], [14, 1, 6]])
    original = raw.copy()
    normalized, raw_norm = normalize_unit_box(raw)
    norm = Normalization(raw_norm.minimum, raw_norm.scale, 10.)
    np.testing.assert_array_equal(norm.inverse_points(normalized), raw)
    np.testing.assert_array_equal(norm.to_mm_points(normalized), raw * 10)
    np.testing.assert_array_equal(norm.from_mm_points(raw * 10), normalized)
    np.testing.assert_array_equal(raw, original)
    assert norm.to_mm_length(1) == 60


def test_cached_header_calibration_never_survives_changed_header(tmp_path):
    path = _stl(tmp_path/'root.stl','[Unit:mm]')
    existing = resolve_batch_units([path],lambda *_a,**_k:pytest.fail('known header prompted'))
    _stl(path,'[Unit:cm]')
    changed = resolve_batch_units([path],lambda *_a,**_k:pytest.fail('known header prompted'),existing=existing)
    assert changed[path]['mm_per_mesh_unit'] == 10
    _stl(path,'unknown')
    with pytest.raises(UnitResolutionCancelled):
        resolve_batch_units([path],lambda *_a,**_k:None,existing=existing)
    with pytest.raises(UnitResolutionError, match='current supported'):
        resolve_unit_calibration(path, 1., source='stl_header')


def test_traits_use_length_area_volume_powers_and_consistent_full_support():
    theta = np.arange(24) * (2*np.pi/24)
    z = np.linspace(0, 5, 15)
    raw = np.c_[np.tile(.2*np.cos(theta),len(z)), np.tile(.2*np.sin(theta),len(z)), np.repeat(z,len(theta))]
    normalized, raw_norm = normalize_unit_box(raw)
    path = raw_norm.transform_points(np.c_[np.zeros(15), np.zeros(15), z])
    mask = np.ones(len(raw), bool)
    first = compute_traits(path, [], normalized, mask, np.zeros(len(raw), int), Normalization(raw_norm.minimum,raw_norm.scale,1.), full_points=raw, full_root_labels=np.zeros(len(raw),int))
    second = compute_traits(path, [], normalized, mask, np.zeros(len(raw), int), Normalization(raw_norm.minimum,raw_norm.scale,10.), full_points=raw, full_root_labels=np.zeros(len(raw),int))
    for column,power in [('length',1),('mean_radius',1),('surface_area',2),('volume',3)]:
        assert second.iloc[0][column] == pytest.approx(first.iloc[0][column] * 10**power)
    assert second.iloc[0].length_unit == 'mm'
    assert second.iloc[0].area_unit == 'mm^2'
    assert second.iloc[0].volume_unit == 'mm^3'


def test_mixed_mm_guide_file_and_raw_endpoints_convert_exactly_once(tmp_path):
    from soyrootbio.pipeline import _manual_primary_guidance, _read_primary_guides
    source=tmp_path/'source.csv'
    source.write_text('x,y,z\n0,0,0\n0,0,5\n',encoding='utf-8')
    guides=tmp_path/'guides.json'
    guides.write_text(json.dumps({'coordinate_unit':'mm','guides':[[120,-70,45]]}),encoding='utf-8')
    config=PipelineConfig(source,tmp_path/'out',start=(12,-7,5),end=(12,-7,4),guide_file=guides,mm_per_mesh_unit=10.)
    snapshot=_manual_primary_guidance(config)
    assert snapshot.coordinate_unit == 'mesh_unit'
    np.testing.assert_array_equal(snapshot.guides,[[12,-7,4.5]])
    config.guidance_coordinate_unit='mm'
    np.testing.assert_array_equal(_read_primary_guides(config),[[120,-70,45]])


@pytest.fixture(scope='module')
def calibrated_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp('physical_bundle')
    # Translation, nonunit scale, and duplicate rows expose frame/order mistakes.
    from soyrootbio.synthetic import write_synthetic_dataset
    path, endpoints = write_synthetic_dataset(directory/'root.csv',primary_points=90,lateral_points=45,lateral_count=2,noise=.002,seed=11)
    raw = pd.read_csv(path).loc[:, ['x','y','z']].to_numpy(float) + np.array([12.,-7.,4.])
    raw = np.vstack([raw,raw[-2:]])
    pd.DataFrame(raw,columns=['x','y','z']).to_csv(path,index=False)
    endpoint_frame = pd.read_csv(endpoints)
    endpoint_frame[['x','y','z']] += np.array([12.,-7.,4.])
    endpoint_frame.to_csv(endpoints,index=False)
    config = PipelineConfig(path,directory/'output',endpoint_file=endpoints,mm_per_mesh_unit=10.,max_root_order=2)
    result = run_pipeline(config)
    return directory,config,result,raw


def test_nonunit_pipeline_exports_all_physical_coordinates_and_original_provenance(calibrated_bundle):
    directory,config,result,raw = calibrated_bundle
    with np.load(bundle_file(config.output_dir, 'original_input_geometry.npz')) as data:
        np.testing.assert_allclose(data['points'],raw,atol=1e-14)
        np.testing.assert_array_equal(data['points'][-2:],data['points'][-4:-2])
    mesh = read_bundle_mesh(config.output_dir)
    np.testing.assert_allclose(mesh.positions,raw*10,atol=1e-12)
    primary = pd.read_csv(config.output_dir/'tables'/'centerlines.csv').query('root_order == 0')
    np.testing.assert_allclose(primary[['x','y','z']], result.normalization.to_mm_points(result.primary_path))
    assert set(primary.coordinate_unit) == {'mm'}
    assert ET.parse(bundle_file(config.output_dir, 'root_system.rsml')).findtext('./metadata/unit') == 'mm'
    with pd.ExcelFile(bundle_file(config.output_dir, 'traits.xlsx')) as xlsx:
        assert set(pd.read_excel(xlsx,sheet_name='root_traits').coordinate_unit) == {'mm'}
    metadata = json.loads((bundle_file(config.output_dir, 'metadata.json')).read_text())
    assert metadata['unit_calibration']['mm_per_mesh_unit'] == 10
    assert metadata['normalization_coordinate_unit'] == 'mesh_unit'
    np.testing.assert_allclose(metadata['normalization_minimum_mm'],np.array(metadata['normalization_minimum'])*10)
    assert metadata['final_centerline_fitting']['unit_context']['coordinate_unit'] == 'analysis_normalized'
    guidance = read_primary_guidance(bundle_file(config.output_dir, 'primary_guidance.json'))
    assert guidance.coordinate_unit == 'mm'
    np.testing.assert_allclose(guidance.start,result.normalization.source_to_output_points(np.array(config.start or pd.read_csv(config.endpoint_file)[['x','y','z']].iloc[0])))


def test_mm_guidance_reload_and_equivalent_raw_cm_encoding_preserve_labels(calibrated_bundle):
    directory,config,result,raw = calibrated_bundle
    rerun = run_pipeline(replace(config,output_dir=directory/'guidance_replay',endpoint_file=bundle_file(config.output_dir, 'primary_guidance.json')))
    np.testing.assert_array_equal(rerun.full_root_labels,result.full_root_labels)
    np.testing.assert_array_equal(rerun.primary_path,result.primary_path)
    equivalent = directory/'same_physical_mm.csv'
    pd.DataFrame(raw*10,columns=['x','y','z']).to_csv(equivalent,index=False)
    endpoint_frame = pd.read_csv(config.endpoint_file)
    endpoint_frame[['x','y','z']] *= 10
    endpoint_path = directory/'equivalent_endpoints.csv'
    endpoint_frame.to_csv(endpoint_path,index=False)
    physical = run_pipeline(replace(config,input_path=equivalent,endpoint_file=endpoint_path,output_dir=directory/'physical_equivalent',mm_per_mesh_unit=1.))
    np.testing.assert_array_equal(physical.full_root_labels,result.full_root_labels)
    for column in ('length','mean_radius','surface_area','volume','mesh_volume','horizontal_reach'):
        np.testing.assert_allclose(physical.traits[column],result.traits[column],rtol=1e-10,equal_nan=True)


def test_new_bundle_editor_recompute_materialize_and_direct_reopen_keep_mm(calibrated_bundle):
    directory,config,result,raw = calibrated_bundle
    session = EditorSession(config.output_dir,session_dir=directory/'editing')
    assert session.coordinate_unit == 'mm'
    top = session.primary_top_reference.copy()
    session._recompute_traits()
    assert session.roots['primary'].traits['length_unit'] == 'mm'
    target = session.export_materialised(directory/'materialized')
    assert json.loads((bundle_file(target, 'edited_root_hierarchy.json')).read_text())['coordinate_unit'] == 'mm'
    assert ET.parse(bundle_file(target, 'edited_root_system.rsml')).findtext('./metadata/unit') == 'mm'
    reopened = EditorSession(target,session_dir=directory/'reopened')
    np.testing.assert_array_equal(reopened.mesh.positions,session.mesh.positions)
    np.testing.assert_array_equal(reopened.primary_top_reference,top)
    assert reopened.coordinate_unit == 'mm'
    current=json.loads((bundle_file(target, 'metadata.json')).read_text())
    assert current['final_compliance_audit']['status'] == 'unresolved_violations_or_missing_native_evidence'
    assert not current['final_compliance_audit']['compliant']
    assert current['higher_order_primary_contact']['status'] == 'unresolved_no_mesh'
    assert current['root_label_map']
    assert current['system_summary']['root_count_total'] == len(session.roots)


def test_shortened_primary_uses_frozen_top_for_both_lateral_orders(calibrated_bundle,tmp_path):
    from soyrootbio.editor.session import RootNode
    directory,config,result,raw=calibrated_bundle
    session=EditorSession(config.output_dir,session_dir=tmp_path/'top-test')
    top=session.primary_top_reference.copy()
    session.roots['primary'].points=np.array([top+[0,0,-10],top+[0,0,-20]])
    first=top+[0,0,-1]
    second=top+[1,0,-2]
    session.roots['o1']=RootNode('o1','primary',1,np.array([first,first+[1,0,-5]]),10,insertion_point=first,insertion_index=0)
    session.roots['o2']=RootNode('o2','o1',2,np.array([second,second+[1,0,-5]]),11,insertion_point=second,insertion_index=0)
    session._validate_changed_lateral_origins({})
    session.roots['o2'].insertion_point=top+[0,0,1]
    with pytest.raises(EditorValidationError,match='above'):
        session._validate_changed_lateral_origins({})
    np.testing.assert_array_equal(session.primary_top_reference,top)


def test_actual_mm_nodule_materialization_and_review_hash_roundtrip(calibrated_bundle,tmp_path):
    from hashlib import sha256
    from soyrootbio.nodules import NoduleResult, geometry_digest, apply_nodule_review
    directory,config,result,raw=calibrated_bundle
    bundle=tmp_path/'nodule-bundle'
    shutil.copytree(config.output_dir,bundle,ignore=shutil.ignore_patterns('.soyrootbio-editor'))
    mesh=read_bundle_mesh(bundle)
    vertices=np.flatnonzero(mesh.root_labels==-1)[:3]
    if len(vertices)<3:
        vertices=np.arange(3)
    fingerprint=sha256(np.sort(vertices).astype('<i8').tobytes()).hexdigest()
    obj={'nodule_id':'nodule-001','numeric_label':-3,'status':'candidate','vertex_indices':vertices.tolist(),
         'geometry_fingerprint':fingerprint,'qc_flags':[],'evidence':{},'review_status':'automatic'}
    (bundle_file(bundle, 'nodules.json')).write_text(json.dumps({'coordinate_unit':'mm','objects':[obj],
        'evidence':{'geometry_sha256':geometry_digest(mesh.positions),'measurement_coordinate_unit':'mm'}}),encoding='utf-8')
    session=EditorSession(bundle,session_dir=tmp_path/'nodule-editor')
    session.apply_operation('review_nodule',{'nodule_id':'nodule-001','status':'rejected'})
    target=session.export_materialised(tmp_path/'nodule-export')
    review=json.loads((bundle_file(target, 'nodule_review.json')).read_text())
    assert review['coordinate_unit'] == 'mm'
    assert json.loads((bundle_file(target, 'nodules.json')).read_text())['coordinate_unit'] == 'mm'
    summary=pd.read_csv(bundle_file(target, 'nodule_summary.csv'))
    assert summary.iloc[0].density_unit == 'mm^-1'
    replay=NoduleResult(np.full(mesh.vertex_count,-1,np.int32),[deepcopy(obj)])
    apply_nodule_review(replay,bundle_file(target, 'nodule_review.json'),mesh.positions,np.zeros(mesh.vertex_count,bool))
    assert replay.objects[0]['status'] == 'rejected'
    assert EditorSession(target,session_dir=tmp_path/'nodule-reopened').coordinate_unit == 'mm'


def test_binary_power_of_two_encoding_is_exact_and_cm_quantization_has_declared_tolerance(tmp_path):
    from soyrootbio.io import load_root_geometry, _mesh_audit
    points=np.array([[12.,-8,4],[13.,-8,4],[12.,-7,4],[12.,-8,6]],dtype=np.float32)
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    points=np.vstack([points+offset for offset in ([0,0,0],[4,0,0],[8,0,0])])
    faces=np.vstack([faces+offset for offset in (0,4,8)])
    def save(path,p,header):
        records=np.zeros(len(faces),dtype=np.dtype([('normal','<f4',(3,)),('points','<f4',(3,3)),('attribute','<u2')]))
        records['points']=p[faces]
        path.write_bytes(header.encode().ljust(80,b'\0')+struct.pack('<I',len(faces))+records.tobytes())
    first=tmp_path/'mm.stl';second=tmp_path/'scale2.stl';cm=tmp_path/'cm.stl'
    save(first,points,'[Unit:mm]');save(second,points/2,'unitless');save(cm,points/10,'[Unit:cm]')
    original=load_root_geometry(first,noise_reduction=False)
    encoded=load_root_geometry(second,noise_reduction=False)
    normalized,norm=normalize_unit_box(original.points)
    scale2,norm2=normalize_unit_box(encoded.points)
    np.testing.assert_array_equal(normalized,scale2)
    np.testing.assert_array_equal(Normalization(norm2.minimum,norm2.scale,2.).to_mm_points(scale2),original.points)
    decoded=load_root_geometry(cm,noise_reduction=False)
    tolerance=4*np.finfo(np.float32).eps*max(1,float(np.max(np.abs(original.export_points))))
    np.testing.assert_allclose(decoded.export_points*10,original.export_points,atol=tolerance,rtol=0)
    raw_audit=_mesh_audit(decoded.export_points,decoded.triangles)
    physical_audit=_mesh_audit(decoded.export_points*10,decoded.triangles)
    assert raw_audit['surface_area_source_units2']*100 == pytest.approx(physical_audit['surface_area_source_units2'],rel=1e-12)
    assert raw_audit['absolute_volume_source_units3']*1000 == pytest.approx(physical_audit['absolute_volume_source_units3'],rel=1e-12)


def test_nonunit_noise_bundle_roundtrip_keeps_raw_provenance_and_mm_measurements(tmp_path):
    from soyrootbio.io import _mesh_audit
    from soyrootbio.export import export_results
    from soyrootbio.presentation import export_noise_free_presentation
    from soyrootbio.noise_bundle import exclude_hidden_noise
    offset=np.array([12.,-7.,4.])
    main=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[0,0,.5]])+offset
    extra=np.array([[.3,0,0],[.31,0,0],[.3,.01,0],[.3,0,.01]])+offset
    raw=np.vstack([main,extra])
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    triangles=np.vstack([faces,faces+4])
    labels=np.array([0]*4+[-1]*4)
    normalized,base=normalize_unit_box(raw)
    norm=Normalization(base.minimum,base.scale,10.)
    primary=norm.transform_points(main[[3,0]])
    traits=compute_traits(primary,[],normalized,labels==0,labels,norm,
        full_points=raw,triangles=triangles,full_root_labels=labels,mesh_metadata=_mesh_audit(raw,triangles))
    source=tmp_path/'native-cm-bundle'
    metadata={'source':'cm_source.stl','coordinate_unit':'mm','unit_calibration':UnitCalibration(10.,'user').to_dict(),
        'd_bar_normalized':.05,'d_bar_mm':.05*norm.output_scale,'normalization_minimum':base.minimum,
        'normalization_scale':base.scale,'normalization_minimum_mm':base.minimum*10,'normalization_scale_mm':base.scale*10,
        'source_geometry':_mesh_audit(raw,triangles),'config':{'tip_vector_window_mm':2.},
        'final_centerline_fitting':{'roots':[{'root_id':'primary','status':'fitted'}],'primary_qc_flags':[]},
        'point_assignment':{'base_point_source_coordinates':(main[3]*10).tolist(),
            'base_tipward_direction':[0,0,-1],'gravity_direction':[0,0,-1],
            'base_collar_neighborhood_radius_normalized':.5,'above_base_tolerance_normalized':1e-9}}
    export_results(source,raw,primary,[],labels==0,labels,traits,norm,metadata,
                   full_points=raw,triangles=triangles,full_root_labels=labels)
    np.savez(bundle_file(source, 'original_input_geometry.npz'),points=raw,triangles=triangles)
    mesh=read_bundle_mesh(source)
    mask=labels==-1
    export_noise_free_presentation(source,mesh.positions,mesh.triangles,colors=mesh.colors/255.,
        root_ids=mesh.root_labels,root_orders=mesh.root_orders,assignment_states=mesh.assignment_states,excluded_mask=mask)
    target=tmp_path/'noise-rebuilt'
    report=exclude_hidden_noise(source,target)
    np.testing.assert_array_equal(read_bundle_mesh(target).positions,raw*10)
    with np.load(bundle_file(target, 'original_input_geometry.npz')) as archive:
        np.testing.assert_array_equal(archive['points'],raw)
    assert set(pd.read_csv(bundle_file(target, 'root_traits.csv')).coordinate_unit)=={'mm'}
    assert report['system_summary']['root_system_surface_area']==pytest.approx(_mesh_audit(main*10,faces)['surface_area_source_units2'])
    assert EditorSession(target,session_dir=tmp_path/'noise-reopened').coordinate_unit=='mm'


def test_mm_hierarchy_correction_uses_physical_transform_for_nonunit_source(tmp_path):
    norm = Normalization(np.array([12.,-5,2]), 5., 10.)
    primary = np.array([[0.,0,1],[0,0,.5],[0,0,0]])
    root = RootPath('child',np.array([[0.,0,.5],[.2,0,.3]]),insertion_point=np.array([0.,0,.5]),insertion_index=1)
    physical = replace(root,points=norm.to_mm_points(root.points),insertion_point=norm.to_mm_points(root.insertion_point[None])[0])
    path = write_editable_hierarchy(tmp_path/'hierarchy.json',norm.to_mm_points(primary),[physical],coordinate_unit='mm')
    restored = apply_hierarchy_corrections(primary,[deepcopy(root)],path,normalization=norm)
    np.testing.assert_allclose(restored[0].points,root.points)
    payload=json.loads(path.read_text())
    payload['roots'][1]['polyline'][1][0] += 2
    path.write_text(json.dumps(payload))
    restored=apply_hierarchy_corrections(primary,[deepcopy(root)],path,normalization=norm)
    assert restored[0].points[1,0] == pytest.approx(root.points[1,0]+.04)


def _widgets(window):
    yield window
    for child in window.winfo_children():
        yield from _widgets(child)


def test_real_hidden_tk_settings_and_unit_dialog_validation_cancel(monkeypatch,tmp_path):
    monkeypatch.setattr(hardware,'detect_gpus',lambda:pytest.fail('GPU detection invoked'))
    root=tk.Tk();root.withdraw()
    try:
        app=BioInsAlgoBatchApp(root)
        texts=[str(widget.cget('text')) for widget in _widgets(root) if 'text' in widget.keys()]
        assert not any('soil' in text.lower() or 'runtime limit' in text.lower() or 'gpu' in text.lower() for text in texts)
        assert all('soil' not in method.lower() for method in PRIMARY_METHODS)
        dialog=UnitScaleDialog(root,tmp_path/'unknown.stl')
        for invalid in ('0','-2','nan','inf','bad'):
            dialog.scale_var.set(invalid);dialog.accept()
            assert dialog.window.winfo_exists()
            assert dialog.result is None
        dialog.scale_var.set('.025');dialog.apply_var.set(True);dialog.accept()
        assert dialog.result == (.025,True)
        cancelled=UnitScaleDialog(root,tmp_path/'unknown.stl');cancelled.cancel()
        assert cancelled.result is None
    finally:
        root.destroy()


def test_selected_primary_sections_have_no_soil_controls_or_text(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from soyrootbio.endpoint_picker import select_primary_sections_gui
    figures=[]
    def show(**kwargs):
        figures.append(plt.gcf())
    monkeypatch.setattr(plt,'show',show)
    with pytest.raises(RuntimeError,match='closed before saving'):
        select_primary_sections_gui(np.array([[0.,0,5],[0,0,0],[1,0,2]]),start=np.array([0,0,5]),end=np.zeros(3))
    labels=[str(artist.get_text()) for artist in figures[0].findobj() if hasattr(artist,'get_text')]
    assert not any('soil' in label.lower() for label in labels)
    assert 'No guides' in labels
    assert 'Save guidance' in labels
    plt.close(figures[0])
