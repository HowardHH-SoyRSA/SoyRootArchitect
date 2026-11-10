"""Audit a completed v2 example, including editor export/reopen and units."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from soyrootbio.bundle_layout import read_bundle_mesh
from soyrootbio.editor.session import EditorSession


METRICS = ('length', 'mean_diameter', 'volume', 'surface_area', 'base_parent_angle_deg',
           'tip_angle_parent_deg', 'tip_primary_angle_deg', 'tip_gravity_angle_deg',
           'tip_start_gravity_angle_deg')
RSML_NUMERIC = (*METRICS, 'volume_frustum', 'mesh_volume')
QUALITY_FIELDS = ('measurement_eligible', 'measurement_identity_resolved', 'measurement_evidence_valid',
                  'measurement_evidence_reason', 'length_estimate_complete', 'radius_profile_complete',
                  'volume_estimate_complete', 'surface_area_estimate_complete',
                  'centerline_measurement_support_complete', 'origin_coverage_complete', 'origin_coverage_reason')
METRIC_FIELDS = tuple(f'{metric}_{suffix}' for metric in METRICS
                     for suffix in ('measurement_status', 'measurement_eligible', 'measurement_reason'))


def _assert_rsml_parity(path, inventory):
    rsml = ET.parse(path)
    nodes = rsml.findall('.//root')
    assert len(nodes) == len(inventory), 'RSML inventory count differs'
    assert {node.attrib['id'] for node in nodes} == set(inventory.root_id)
    fields = (*RSML_NUMERIC, *QUALITY_FIELDS, *METRIC_FIELDS)
    for row in inventory.itertuples():
        node = rsml.find(f".//root[@id='{row.root_id}']/properties")
        assert node is not None, row.root_id
        for key in fields:
            value, item = getattr(row, key), node.find(key)
            assert item is not None, (row.root_id, key, 'missing RSML property')
            # These two legacy flags intentionally serialize an unavailable
            # completeness flag as false rather than a nonfinite boolean.
            if key in {'length_estimate_complete', 'radius_profile_complete'} and pd.isna(value):
                value = False
            if pd.isna(value):
                assert item.attrib.get('available') == 'false' and 'value' not in item.attrib, (row.root_id, key)
            elif isinstance(value, (bool, np.bool_)):
                assert item.attrib.get('value') == str(bool(value)).lower(), (row.root_id, key)
            elif isinstance(value, (int, float, np.number)):
                assert 'value' in item.attrib and item.attrib.get('available') != 'false', (row.root_id, key)
                np.testing.assert_allclose(float(item.attrib['value']), value, rtol=1e-12, atol=1e-12,
                                           err_msg=f'{row.root_id}: {key}')
            else:
                assert item.attrib.get('value') == str(value), (row.root_id, key)
    return list(fields)


def _assert_frame_parity(source, result, columns):
    assert source.index.equals(result.index), 'round-trip table identities differ'
    for key in columns:
        assert key in source and key in result, key
        if pd.api.types.is_numeric_dtype(source[key]) and not pd.api.types.is_bool_dtype(source[key]):
            np.testing.assert_allclose(source[key], result[key], rtol=1e-8, atol=1e-8, equal_nan=True, err_msg=key)
        else:
            pd.testing.assert_series_equal(source[key], result[key], check_names=False, check_dtype=False,
                                           obj=key)


def main(bundle, work):
    assert not work.resolve().is_relative_to(bundle.resolve()), 'validation work must be outside the immutable bundle'
    work.mkdir(parents=True,exist_ok=True)
    table_dir=bundle/'tables';support=bundle/'support'
    roots=pd.read_csv(table_dir/'root_traits.csv');inventory=pd.read_csv(table_dir/'root_inventory.csv')
    summary=pd.read_csv(table_dir/'architecture_summary.csv').set_index('scope')
    mapping=pd.read_csv(table_dir/'root_label_map.csv')
    metadata=json.loads((support/'metadata.json').read_text(encoding='utf-8'))
    hierarchy=json.loads((support/'root_hierarchy.json').read_text(encoding='utf-8'))
    mesh=read_bundle_mesh(bundle)
    assert roots.root_id.is_unique and inventory.root_id.is_unique
    assert set(roots.root_id)==set(inventory.loc[inventory.measurement_eligible.eq(True),'root_id'])
    assert set(inventory.root_id)=={r['root_id'] for r in hierarchy['roots']}
    assert (np.isfinite(roots.length)&(roots.length>0)).all()
    for path in bundle.rglob('*'):
        if not path.is_file():continue
        if path.suffix in {'.csv','.xlsx'}:assert path.parent==table_dir
        elif path.suffix not in {'.png','.ply'}:assert support in path.parents
    assert not (bundle/'segmented_root_structure.ply').exists()
    from soyrootbio.editor.ply import read_labeled_ply
    presentation=read_labeled_ply(bundle/'presentation_root_structure.ply')
    with np.load(support/'presentation_vertex_mapping.npz') as data:indices=data['presentation_to_source']
    np.testing.assert_array_equal(presentation.positions,mesh.positions[indices])
    np.testing.assert_array_equal(presentation.root_labels,mesh.root_labels[indices])
    with np.load(support/'presentation_noise_masks.npz') as data:noise=data['excluded_full_vertices']
    np.testing.assert_array_equal(indices,np.flatnonzero(~noise))
    assert (mesh.root_labels[noise]==-1).all()
    source_mapping_status = 'unavailable_no_input_index_map'
    if (support/'input_geometry_mapping.npz').exists():
        with np.load(support/'original_input_geometry.npz') as raw,np.load(support/'input_geometry_mapping.npz') as data:
            original_to_full=data['original_to_full'];valid=original_to_full>=0
            factor=metadata['unit_calibration']['mm_per_mesh_unit']
            np.testing.assert_allclose(mesh.positions[original_to_full[valid]],raw['points'][valid]*factor,rtol=0,atol=1e-12)
        source_mapping_status = 'exact_for_mapped_input_vertices'
    book=load_workbook(table_dir/'traits.xlsx',data_only=True,read_only=True)
    for path in table_dir.glob('*.csv'):
        frame=pd.read_csv(path)
        sheet=book[path.stem[:31]]
        values=list(sheet.values)
        assert list(values[0])==frame.columns.tolist(),path.name
        assert len(values)-1==len(frame),path.name
        for csv_row,xlsx_row in zip(frame.itertuples(index=False,name=None),values[1:]):
            for a,b in zip(csv_row,xlsx_row):
                if pd.isna(a):assert b is None,(path.name,a,b)
                elif isinstance(b,bool):assert str(a).lower()==str(b).lower(),(path.name,a,b)
                elif isinstance(b,(int,float)):assert np.isclose(float(a),b,rtol=1e-10,atol=1e-10),(path.name,a,b)
                else:assert str(a)==str(b),(path.name,a,b)
    book.close()
    # Independently integrate the selected union via surface centroid flux.
    surface_ids = (inventory.loc[inventory.surface_area_measurement_eligible.eq(True), 'root_id']
                   if 'surface_area_measurement_eligible' in inventory else roots.root_id)
    eligible=mapping[mapping.root_id.isin(surface_ids)].numeric_label.to_numpy(int)
    f=mesh.triangles[np.all(np.isin(mesh.root_labels[mesh.triangles],eligible),axis=1)]
    ids,inv=np.unique(f,return_inverse=True);f=inv.reshape(-1,3);xyz=mesh.positions[ids]
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(xyz),len(xyz)))
    _,components=connected_components(graph,directed=False);_,first=np.unique(components,return_index=True)
    fc=components[f[:,0]]; origin=xyz[first][fc]
    a,b,c=(xyz[f[:,i]] for i in range(3));normal=np.cross(b-a,c-a)
    flux=np.einsum('ij,ij->i',(a+b+c)/3-origin,normal)/6
    mesh_volume=np.abs(np.bincount(fc,weights=flux)).sum() if len(f) else np.nan
    area=.5*np.linalg.norm(normal,axis=1).sum()
    np.testing.assert_allclose([mesh_volume,area],[summary.loc['system','mesh_volume'],summary.loc['system','mesh_surface_area']],rtol=1e-10,equal_nan=True)
    if not len(f):
        assert summary.loc['system','mesh_volume_reliable'] == False
    volume_rows = (inventory.loc[inventory.volume_measurement_eligible.eq(True)]
                   if 'volume_measurement_eligible' in inventory else roots)
    np.testing.assert_allclose(volume_rows.volume_frustum.sum(min_count=1),summary.loc['system','volume_frustum'],rtol=1e-12,equal_nan=True)
    rsml_fields = _assert_rsml_parity(support/'root_system.rsml', inventory)
    before={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
    session=EditorSession(bundle,session_dir=work/'editor_session')
    exported=session.export_materialised(work/'materialized')
    reopened=EditorSession(exported,session_dir=work/'editor_reopen')
    np.testing.assert_array_equal(session.mesh.positions,reopened.mesh.positions)
    np.testing.assert_array_equal(session.mesh.triangles,reopened.mesh.triangles)
    np.testing.assert_array_equal(session.mesh.root_labels,reopened.mesh.root_labels)
    np.testing.assert_array_equal(session.primary_top_reference,reopened.primary_top_reference)
    assert set(session.roots)==set(reopened.roots)==set(inventory.root_id)
    modern_inventory = 'length_measurement_status' in inventory
    remeasured=pd.read_csv(exported/'tables'/('root_inventory.csv' if modern_inventory else 'root_traits.csv')).set_index('root_id').sort_index()
    source=(inventory if modern_inventory else roots).set_index('root_id').sort_index()
    roundtrip_fields = list(dict.fromkeys((*RSML_NUMERIC, *QUALITY_FIELDS, *METRIC_FIELDS,
                                          'horizontal_reach', 'chord_length', 'tortuosity')))
    _assert_frame_parity(source, remeasured, roundtrip_fields)
    _assert_rsml_parity(exported/'support/root_system.rsml', remeasured.reset_index())
    exported_summary = pd.read_csv(exported/'tables/architecture_summary.csv').set_index('scope').sort_index()
    source_summary = summary.sort_index()
    _assert_frame_parity(source_summary, exported_summary, source_summary.columns)
    after={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
    assert before==after
    report={'passed':True,'eligible_roots':len(roots),'excluded_roots':len(inventory)-len(roots),
            'topology_roots':len(inventory),'noise_vertices':int(noise.sum()),
            'csv_workbook_parity':True,'rsml_measurement_parity':True,
            'rsml_checked_properties':rsml_fields, 'editor_roundtrip_checked_columns':roundtrip_fields,
            'architecture_summary_roundtrip_columns':list(source_summary.columns),
            'source_geometry_mapping_status':source_mapping_status,'presentation_mapping_exact':True,
            'independent_mesh_flux_volume':float(mesh_volume) if np.isfinite(mesh_volume) else None,
            'mesh_volume_reported':float(summary.loc['system','mesh_volume']) if np.isfinite(summary.loc['system','mesh_volume']) else None,
            'mesh_flux_is_anatomical_volume_certificate':False,
            'editor_geometry_labels_top_traits_roundtrip':True,'automatic_bundle_unchanged':True,
            'final_native_audit':metadata['final_compliance_audit']}
    (work/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();main(args.bundle.resolve(),args.work.resolve())
