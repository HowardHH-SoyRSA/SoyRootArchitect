import json
import numpy as np
import pytest

from soyrootbio.io import write_labeled_ply
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.presentation import NoisePresentation, export_noise_free_presentation


def sample(tmp_path):
    points = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [2.,0,0],[2.1,0,0],[2,.1,0],[2,0,.1]])
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3],
                      [4,6,5],[4,5,7],[5,6,7],[6,4,7]])
    labels = np.array([0]*4+[-1]*4)
    path = tmp_path/'segmented_root_structure.ply'
    write_labeled_ply(path,points,triangles=faces,root_ids=labels)
    return read_labeled_ply(path)


def export(tmp_path,mesh,mask):
    return export_noise_free_presentation(
        tmp_path,mesh.positions,mesh.triangles,colors=mesh.colors/255.,
        root_ids=mesh.root_labels,root_orders=mesh.root_orders,
        assignment_states=mesh.assignment_states,excluded_mask=mask,
        review={'reason':'reviewed exterior fragment'},
    )


def test_compact_ply_and_editor_display_preserve_source_indices_and_traits(tmp_path):
    mesh=sample(tmp_path);before=mesh.path.read_bytes()
    mask=np.array([False]*4+[True]*4)
    report=export(tmp_path,mesh,mask)
    clean=read_labeled_ply(tmp_path/'presentation_root_structure.ply')
    with np.load(tmp_path/'presentation_vertex_mapping.npz') as data:
        ids=data['presentation_to_source']
        np.testing.assert_array_equal(clean.positions,mesh.positions[ids])
        np.testing.assert_array_equal(ids[clean.triangles],mesh.triangles[:4])
    assert report['additional_review_hidden_vertex_count']==0
    assert report['analysis_excluded_vertex_count']==4
    assert report['all_hidden_vertices_excluded_from_assignment_and_traits']
    assert not report['additional_review_changes_analysis_or_traits']
    presentation=NoisePresentation(tmp_path,mesh)
    assert presentation.hidden_labels()==set()
    session=tmp_path/'session';session.mkdir()
    display=read_labeled_ply(presentation.mesh_path(session))
    np.testing.assert_array_equal(display.positions,mesh.positions)
    np.testing.assert_array_equal(display.root_labels,mesh.root_labels)
    np.testing.assert_array_equal(display.triangles,mesh.triangles[:4])
    assert before==mesh.path.read_bytes()
    assert not (tmp_path/'root_traits.csv').exists()


def test_presentation_rejects_wrong_geometry_and_partial_component(tmp_path):
    mesh=sample(tmp_path);mask=np.array([False]*4+[True]*4)
    export(tmp_path,mesh,mask)
    mesh.positions[0,0]+=.01
    with pytest.raises(ValueError,match='different geometry'):
        NoisePresentation(tmp_path,mesh)
    with pytest.raises(ValueError,match='cut a connected'):
        export(tmp_path,mesh,np.array([False]*7+[True]))


def test_legacy_analysis_mask_only_hides_unassigned_noise(tmp_path):
    mesh=sample(tmp_path);mask=np.array([False]*4+[True]*4)
    mesh.root_labels[mask]=1
    np.savez(tmp_path/'noise_reduction_masks.npz',excluded_full_vertices=mask)
    with pytest.raises(ValueError,match='inconsistent'):
        NoisePresentation(tmp_path,mesh)
    mesh.root_labels[mask]=-1
    display=NoisePresentation(tmp_path,mesh)
    assert display.hidden_labels()==set()
    np.testing.assert_array_equal(display.mask,mask)


def test_display_only_exclusions_and_assigned_hidden_vertices_are_rejected(tmp_path):
    mesh=sample(tmp_path);mask=np.array([False]*4+[True]*4)
    with pytest.raises(ValueError,match='excluded from analysis'):
        export_noise_free_presentation(tmp_path,mesh.positions,mesh.triangles,
            colors=mesh.colors/255.,root_ids=mesh.root_labels,root_orders=mesh.root_orders,
            assignment_states=mesh.assignment_states,excluded_mask=mask,analysis_noise_mask=np.zeros(8,bool))
    mesh.root_labels[mask]=1
    with pytest.raises(ValueError,match='unassigned'):
        export(tmp_path,mesh,mask)
