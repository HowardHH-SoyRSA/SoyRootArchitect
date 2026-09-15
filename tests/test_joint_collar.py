import json
import itertools

import numpy as np
import pytest

from soyrootbio.collar import analyze_joint_collar
from soyrootbio.types import RootPath


def crown():
    """Tapered trunk and three separately triangulated emerging cylinders."""
    points, faces, labels, paths, groups = [], [], [], [], []
    theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
    def tube(line, radial, label):
        offset = len(points)
        groups.append(np.arange(offset, offset+len(line)*24))
        points.extend((line[:,None,:]+radial).reshape(-1,3))
        labels.extend([label]*len(line)*24)
        for i in range(len(line)-1):
            for j in range(24):
                a, b = offset+i*24+j, offset+i*24+(j+1)%24
                faces.extend([[a,b,a+24],[b,b+24,a+24]])
    z = np.linspace(0, .8, 161)
    radius = .045-.02*z/.8
    tube(np.c_[z*0,z*0,z], np.stack([radius[:,None]*np.cos(theta),
        radius[:,None]*np.sin(theta),np.zeros((len(z),24))],axis=2),0)
    for label, (angle, height) in enumerate([(0,.06),(2.1,.085),(4.2,.11)],1):
        x = np.linspace(0,.32,65)
        axis = np.array([np.cos(angle),np.sin(angle),0.])
        normal = np.array([-np.sin(angle),np.cos(angle),0.])
        line = x[:,None]*axis + [0,0,height]
        radial = .009*(np.cos(theta)[:,None]*normal + np.sin(theta)[:,None]*[0,0,1])
        tube(line,radial,label)
        paths.append(RootPath(root_id=f"child-{label}", parent_id="primary", order=1,
            points=line[[0,8,-1]].copy(), body_start_index=1))
    return np.array(points),np.array(labels),np.array([[0.,0.,0.],[0.,0.,.8]]),paths,np.array(faces),groups


def run(case, **kwargs):
    p, labels, primary, roots, faces, _ = case
    result, report = analyze_joint_collar(p,labels,primary,roots,d_bar=.004,triangles=faces,**kwargs)
    json.dumps(report,allow_nan=False)
    return result,report


def corrupted_crown():
    case = crown()
    p, labels, _, _, _, groups = case
    for group in groups[1:]:
        radial = np.linalg.norm(p[group,:2],axis=1)
        labels[group[(radial>.055)&(radial<.10)]] = 0
        labels[group[radial<.035]] = 0
    trunk = groups[0]
    band = trunk[(p[trunk,2]>.18)&(p[trunk,2]<.24)]
    labels[band] = np.where(p[band,0]>0,1,2)
    return case


def test_joint_recovers_three_emergences_and_lateral_spread_on_trunk():
    case = corrupted_crown()
    p, labels, _, roots, _, groups = case
    original_paths = [r.points.copy() for r in roots]
    after, report = run(case)
    assert len(report["participating_roots"]) == 4
    trunk = groups[0]
    band = trunk[(p[trunk,2]>.18)&(p[trunk,2]<.24)]
    assert np.all(after[band] == 0)
    for label,group in enumerate(groups[1:],1):
        radius = np.linalg.norm(p[group,:2],axis=1)
        emergence = group[(radius>.06)&(radius<.09)]
        assert np.all(after[emergence] == label)
        # The internal connector is not an exposed child surface.
        assert np.all(after[group[radius<.035]] == 0)
        body = group[radius>.1]
        np.testing.assert_array_equal(after[body],labels[body])
    assert report["changed_vertex_count"] > 0
    for root,path in zip(roots,original_paths):
        np.testing.assert_array_equal(root.points,path)


def test_all_root_permutations_produce_identical_ownership_and_topology():
    case = corrupted_crown()
    expected, first = run(case)
    for permutation in itertools.permutations(range(3)):
        remap = np.array([0]+[i+1 for i in permutation])
        inverse = np.argsort(remap)
        labels = case[1].copy()
        labels[labels>=0] = inverse[labels[labels>=0]]
        reordered = (*case[:1],labels,case[2],[case[3][i] for i in permutation],*case[4:])
        result, report = run(reordered)
        result[result>=0] = remap[result[result>=0]]
        np.testing.assert_array_equal(result,expected)
        assert report["topology_decisions"] == first["topology_decisions"]


def test_shoot_side_all_assignment_states_are_unassigned_and_barriers():
    case = corrupted_crown()
    excluded = case[0][:,2]<.04
    case[1][np.flatnonzero(excluded)[::2]] = -2
    result,report = run(case,excluded_mask=excluded)
    assert np.all(result[excluded]==-1)
    assert report["excluded_to_unassigned_count"] == excluded.sum()


def test_disconnected_emergence_cannot_use_spatially_close_seeds():
    case = list(corrupted_crown())
    p, _, _, _, faces, groups = case
    radius = np.linalg.norm(p[:,:2],axis=1)
    for group in groups[1:]:
        case[1][group[radius[group]<.10]] = 0
    child_face = np.isin(faces[:,0],np.concatenate(groups[1:]))
    crossing = (radius[faces].min(axis=1)<.10)&(radius[faces].max(axis=1)>=.10)
    case[4] = faces[~(child_face & crossing)]
    result, report = run(case)
    for group in groups[1:]:
        region = group[(radius[group]>.06)&(radius[group]<.09)]
        assert np.all(result[region]==0)
    assert report["unresolved_vertex_count"] > 0


def test_adaptive_taper_density_and_scale_not_fixed_global_radius():
    case = corrupted_crown()
    result,report = run(case)
    profile = report["local_profiles"]["primary"]
    assert np.ptp(profile["radius"])>.008
    assert np.ptp(profile["spacing"])>0
    assert report["bounds"]["primary_arc_max"]>.25
    scaled = list(corrupted_crown())
    scaled[0] *= 13
    scaled[2] *= 13
    for r in scaled[3]:
        r.points *= 13
    again, _ = analyze_joint_collar(*scaled[:4],d_bar=.052,triangles=scaled[4])
    np.testing.assert_array_equal(again,result)


def test_equal_sibling_hypotheses_keep_supported_surfaces_and_report_overlap():
    case = list(crown())
    group = case[5][1]
    original = case[1].copy()
    case[3].append(RootPath(root_id="duplicate",points=case[3][0].points.copy(),
                           parent_id="primary",order=1,body_start_index=1))
    case[1][group[::2]] = 4
    after,report = run(case)
    region = group[np.linalg.norm(case[0][group,:2],axis=1)>.06]
    assert np.all(after[region]>0)
    assert any(r["reason"]=="joint_score_tie" for r in report["unresolved_regions"])
    assert np.all(original[group]==1)


def test_higher_order_and_unsupported_roots_participate_without_inventing_parents():
    case = crown()
    case[3][1].parent_id="child-1"
    case[3][1].order=2
    case[1][case[1]==3]=0
    _,report = run(case)
    assert {r["root_id"] for r in report["participating_roots"]} == {"primary","child-1","child-2","child-3"}
    row = next(r for r in report["topology_decisions"] if r["root_id"]=="child-3")
    assert row["status"]=="unresolved_emergence"
    assert all(r["action"]=="preserve" for r in report["topology_decisions"])


def test_no_mesh_keeps_labels_and_records_missing_support():
    case = corrupted_crown()
    result, report = analyze_joint_collar(*case[:4],d_bar=.004)
    np.testing.assert_array_equal(result,case[1])
    assert report["connectivity"]=="no_mesh_support"


def test_default_boundary_excludes_shoot_side_without_a_supplied_mask():
    case = list(crown())
    # Extend the first trunk rings above the primary start cross-section.
    case[0][case[5][0][:5*24],2] -= .04
    after, report = run(case)
    shoot = case[0][:,2]<-.01
    assert shoot.any() and np.all(after[shoot]==-1)
    assert report["excluded_to_unassigned_count"]>0


def test_single_station_nearby_root_is_included_and_flagged():
    case = crown()
    case[3][0].points = case[3][0].points[:1]
    case[3][0].body_start_index = 0
    _, report = run(case)
    assert "child-1" in {r["root_id"] for r in report["participating_roots"]}
    row = next(r for r in report["topology_decisions"] if r["root_id"]=="child-1")
    assert row["status"]=="unresolved_emergence"


def test_collinear_resampling_does_not_change_the_collar_decision():
    case = corrupted_crown()
    expected, _ = run(case)
    for root in case[3]:
        root.points = np.linspace(root.points[0],root.points[-1],65)
        root.body_start_index = 8
    case = (case[0],case[1],np.linspace(case[2][0],case[2][-1],161),*case[3:])
    actual, _ = run(case)
    np.testing.assert_array_equal(actual,expected)


def test_joint_transfer_cannot_cut_a_donor_surface_bridge():
    case = crown()
    group = case[5][1]
    radius = np.linalg.norm(case[0][group,:2],axis=1)
    case[1][group[radius<.18]] = 0
    # An exposed child anchor occupies only half a ring. Primary ownership
    # still connects both sides through the other half of this surface.
    seeds = (radius>.10)&(radius<.12)&(case[0][group,1]>0)
    case[1][group[seeds]] = 1
    after,report = run(case)
    assert any(r["reason"]=="donor_surface_bridge_would_be_cut" for r in report["unresolved_regions"])
    for row in report["unresolved_regions"]:
        if row["reason"]=="donor_surface_bridge_would_be_cut":
            np.testing.assert_array_equal(after[row["vertex_indices"]],case[1][row["vertex_indices"]])


@pytest.mark.parametrize("spacing",[0,float("nan"),-1])
def test_rejects_invalid_sampling(spacing):
    with pytest.raises(ValueError,match="d_bar"):
        analyze_joint_collar(*crown()[:4],d_bar=spacing)
