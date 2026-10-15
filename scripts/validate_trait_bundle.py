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


def main(bundle, work):
    work.mkdir(parents=True,exist_ok=True)
    table_dir=bundle/'tables';support=bundle/'support'
    roots=pd.read_csv(table_dir/'root_traits.csv');inventory=pd.read_csv(table_dir/'root_inventory.csv')
    summary=pd.read_csv(table_dir/'architecture_summary.csv').set_index('scope')
    mapping=pd.read_csv(table_dir/'root_label_map.csv')
    metadata=json.loads((support/'metadata.json').read_text(encoding='utf-8'))
    hierarchy=json.loads((support/'root_hierarchy.json').read_text(encoding='utf-8'))
    mesh=read_bundle_mesh(bundle)
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
    with np.load(support/'original_input_geometry.npz') as raw,np.load(support/'input_geometry_mapping.npz') as data:
        original_to_full=data['original_to_full'];valid=original_to_full>=0
        factor=metadata['unit_calibration']['mm_per_mesh_unit']
        np.testing.assert_allclose(mesh.positions[original_to_full[valid]],raw['points'][valid]*factor,rtol=0,atol=1e-12)
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
    eligible=mapping[mapping.root_id.isin(roots.root_id)].numeric_label.to_numpy(int)
    f=mesh.triangles[np.all(np.isin(mesh.root_labels[mesh.triangles],eligible),axis=1)]
    ids,inv=np.unique(f,return_inverse=True);f=inv.reshape(-1,3);xyz=mesh.positions[ids]
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(xyz),len(xyz)))
    _,components=connected_components(graph,directed=False);_,first=np.unique(components,return_index=True)
    fc=components[f[:,0]]; origin=xyz[first][fc]
    a,b,c=(xyz[f[:,i]] for i in range(3));normal=np.cross(b-a,c-a)
    flux=np.einsum('ij,ij->i',(a+b+c)/3-origin,normal)/6
    mesh_volume=np.abs(np.bincount(fc,weights=flux)).sum()
    area=.5*np.linalg.norm(normal,axis=1).sum()
    np.testing.assert_allclose([mesh_volume,area],[summary.loc['system','mesh_volume'],summary.loc['system','mesh_surface_area']],rtol=1e-10)
    np.testing.assert_allclose(roots.volume_frustum.sum(),summary.loc['system','volume_frustum'],rtol=1e-12)
    rsml=ET.parse(support/'root_system.rsml')
    for row in roots.itertuples():
        node=rsml.find(f".//root[@id='{row.root_id}']/properties")
        for key in ('length','volume_frustum','mesh_volume'):
            val=getattr(row,key); item=node.find(key)
            if np.isfinite(val):assert np.isclose(float(item.attrib['value']),val,rtol=1e-12,atol=1e-12)
            else:assert item.attrib['available']=='false'
    before={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
    session=EditorSession(bundle,session_dir=work/'editor_session')
    exported=session.export_materialised(work/'materialized')
    reopened=EditorSession(exported,session_dir=work/'editor_reopen')
    np.testing.assert_array_equal(session.mesh.positions,reopened.mesh.positions)
    np.testing.assert_array_equal(session.mesh.triangles,reopened.mesh.triangles)
    np.testing.assert_array_equal(session.mesh.root_labels,reopened.mesh.root_labels)
    np.testing.assert_array_equal(session.primary_top_reference,reopened.primary_top_reference)
    assert set(session.roots)==set(reopened.roots)==set(inventory.root_id)
    remeasured=pd.read_csv(exported/'tables/root_traits.csv').set_index('root_id').sort_index()
    source=roots.set_index('root_id').sort_index()
    for key in ('length','mean_diameter','volume_frustum','mesh_volume','horizontal_reach'):
        np.testing.assert_allclose(source[key],remeasured[key],rtol=1e-8,atol=1e-8,equal_nan=True)
    after={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
    assert before==after
    report={'passed':True,'eligible_roots':len(roots),'excluded_roots':len(inventory)-len(roots),
            'topology_roots':len(inventory),'noise_vertices':int(noise.sum()),
            'csv_workbook_parity':True,'rsml_measurement_parity':True,
            'source_geometry_mapping_exact':True,'presentation_mapping_exact':True,
            'independent_mesh_flux_volume':float(mesh_volume),'mesh_volume_reported':float(summary.loc['system','mesh_volume']),
            'editor_geometry_labels_top_traits_roundtrip':True,'automatic_bundle_unchanged':True,
            'final_native_audit':metadata['final_compliance_audit']}
    (work/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();main(args.bundle.resolve(),args.work.resolve())
