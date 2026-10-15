from __future__ import annotations

import numpy as np
import pytest

from soyrootbio.traits import _radius_profile, _frustum_measurements, compute_traits
from soyrootbio.types import Normalization


def _cylinder(z, radius=.2):
    theta = np.arange(24)*2*np.pi/24
    z = np.asarray(z)
    return np.c_[np.tile(radius*np.cos(theta),len(z)),np.tile(radius*np.sin(theta),len(z)),np.repeat(z,len(theta))]


def _path(z):
    return np.c_[np.zeros(len(z)),np.zeros(len(z)),z]


def test_added_distant_collinear_continuation_cannot_inflate_terminal_radius_or_volume():
    support=_cylinder(np.linspace(0,5,31))
    path=_path(np.linspace(0,5,21))
    norm=Normalization(np.zeros(3),1.,1.)
    first,second={},{}
    expected=_radius_profile(support,path,norm,assessment=first)
    extra=_cylinder(np.linspace(10,75,60))
    before=extra.copy()
    actual=_radius_profile(np.vstack([support,extra]),path,norm,assessment=second)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual,.2,atol=1e-12)
    np.testing.assert_array_equal(extra,before)
    assert second['outside_body_support_count'] == len(extra)
    assert second['body_coverage_complete']
    assert _frustum_measurements(path,actual)[1] == pytest.approx(np.pi*.2**2*5)


def test_support_inside_terminal_slab_uses_transverse_distance():
    support=_cylinder(np.linspace(0,5,31))
    support=np.vstack([support,_cylinder([5.025])])
    assessment={}
    radii=_radius_profile(support,_path(np.linspace(0,5,21)),Normalization(np.zeros(3),1.),assessment=assessment)
    np.testing.assert_allclose(radii,.2,atol=1e-12)
    assert assessment['outside_body_support_count'] == 0


def test_empty_and_one_section_support_remain_explicit_incomplete_estimates():
    path=_path(np.linspace(0,5,21))
    norm=Normalization(np.zeros(3),1.,1.)
    assessment={}
    empty=_radius_profile(np.empty((0,3)),path,norm,assessment=assessment)
    assert np.isnan(empty).all()
    assert not assessment['body_coverage_complete']
    support=_cylinder([2.5])
    measured=compute_traits(path,[],support,np.ones(len(support),bool),np.zeros(len(support),int),norm)
    row=measured.iloc[0]
    assert np.isfinite(row.volume)
    assert not row.volume_estimate_complete
    assert not row.body_coverage_complete
    assert 'radius_endpoint_support_incomplete' in row.qc_flags
    assert measured.attrs['system_summary']['root_system_volume_estimate_partial_root_count'] == 1
    assert not measured.attrs['system_summary']['root_system_volume_estimate_complete']


def test_duplicate_positions_do_not_change_sampling_slab_or_measurements():
    support=_cylinder(np.linspace(0,5,31))
    duplicated=np.repeat(support,20,axis=0)
    before=duplicated.copy()
    path=_path(np.linspace(0,5,21))
    first,second={},{}
    normal=_radius_profile(support,path,Normalization(np.zeros(3),1.),assessment=first)
    copies=_radius_profile(duplicated,path,Normalization(np.zeros(3),1.),assessment=second)
    np.testing.assert_allclose(copies,normal,rtol=1e-12,atol=1e-12)
    assert second['sampling_spacing'] == pytest.approx(first['sampling_spacing'])
    np.testing.assert_array_equal(duplicated,before)


def test_scaled_translated_source_has_same_physical_radius_coverage_and_volume():
    support=_cylinder(np.linspace(0,5,31))
    path=_path(np.linspace(0,5,21))
    expected=_radius_profile(support,path,Normalization(np.zeros(3),1.,1.))
    offset=np.array([12.,-7.,4.])
    norm=Normalization(offset,.5,10.)
    raw_support=support/10+offset
    raw_path=path/10+offset
    actual=_radius_profile(norm.transform_points(raw_support),norm.transform_points(raw_path),norm)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert _frustum_measurements(norm.to_mm_points(norm.transform_points(raw_path)),actual)[1] == pytest.approx(np.pi*.2**2*5)


def test_curve_resampling_keeps_geometric_coverage_and_radius_consistent():
    angle=np.linspace(0,.6,101)
    theta=np.arange(24)*2*np.pi/24
    center=np.c_[10*np.sin(angle),10*(1-np.cos(angle)),np.zeros(len(angle))]
    normal=np.c_[np.sin(angle),-np.cos(angle),np.zeros(len(angle))]
    support=(center[:,None,:]+.2*np.cos(theta)[None,:,None]*normal[:,None,:]
             +.2*np.sin(theta)[None,:,None]*np.array([0,0,1])).reshape(-1,3)
    means=[]
    for count in (41,161):
        a=np.linspace(0,.6,count)
        path=np.c_[10*np.sin(a),10*(1-np.cos(a)),np.zeros(len(a))]
        assessment={}
        radii=_radius_profile(support,path,Normalization(np.zeros(3),1.),assessment=assessment)
        means.append(radii.mean())
        assert assessment['body_coverage_complete']
        assert assessment['outside_body_support_count'] == 0
    assert means[0] == pytest.approx(means[1],rel=.01)


def test_tapered_surface_volume_is_consistent_when_identical_line_is_densified():
    z=np.linspace(0,5,31)
    theta=np.arange(24)*2*np.pi/24
    r=.1+.04*z
    support=np.c_[(r[:,None]*np.cos(theta)).ravel(),(r[:,None]*np.sin(theta)).ravel(),np.repeat(z,len(theta))]
    analytic=np.pi*5*(.1**2+.1*.3+.3**2)/3
    volumes=[]
    for count in (21,41,161,401):
        path=_path(np.linspace(0,5,count))
        assessment={}
        radii=_radius_profile(support,path,Normalization(np.zeros(3),1.),assessment=assessment)
        volumes.append(_frustum_measurements(path,radii)[1])
        assert assessment['body_coverage_complete']
        assert not assessment['global_radius_fallback']
        assert volumes[-1] == pytest.approx(analytic,rel=.03)
    assert max(volumes)/min(volumes)-1 <= .02


def test_nonuniform_taper_stations_do_not_bias_exported_mean_or_median_diameter():
    z=np.linspace(0,5,301)
    theta=np.arange(24)*2*np.pi/24
    r=.1+.04*z
    support=np.c_[(r[:,None]*np.cos(theta)).ravel(),(r[:,None]*np.sin(theta)).ravel(),np.repeat(z,len(theta))]
    means,medians=[],[]
    for power in (.5,1.,2.):
        path=_path(5*np.linspace(0,1,201)**power)
        measured=compute_traits(path,[],support,np.ones(len(support),bool),np.zeros(len(support),int),Normalization(np.zeros(3),1.,1.))
        row=measured.iloc[0]
        means.append(row.mean_diameter);medians.append(row.median_diameter)
        assert row.diameter_statistics_method == 'arc-weighted-piecewise-linear-radius-v1'
        assert row.mean_diameter == pytest.approx(.4,rel=.015)
        assert row.median_diameter == pytest.approx(.4,rel=.015)
    assert max(means)/min(means)-1 < .02
    assert max(medians)/min(medians)-1 < .02


def test_unavailable_primary_summary_does_not_turn_unknown_into_zero():
    support=_cylinder(np.linspace(0,5,31))
    traits=compute_traits(_path(np.linspace(0,5,21)),[],support,np.ones(len(support),bool),np.zeros(len(support),int),
                          Normalization(np.zeros(3),1.,1.),primary_qc_flags=['centerline_no_support'])
    summary=traits.attrs['system_summary']
    assert np.isnan(summary['primary_root_length'])
    assert np.isnan(summary['primary_root_volume_estimate'])
    assert np.isnan(summary['root_system_length'])
    assert summary['lateral_root_length_sum'] == 0
    assert not traits.iloc[0].radius_profile_complete
    assert np.isnan(traits.iloc[0].mean_radius)
    assert np.isnan(traits.iloc[0].mean_diameter)


@pytest.mark.parametrize('assessment', [
    {'status':'partial_support'}, {'status':'qc_rejected_prior_retained'},
    {'parent_connector_mode':'unsupported_gap'}, {'parent_connector_mode':'preserved_topology'},
    {'exposed_body_supported':False}, {'fit_qc_passed':False}, {'traits_geometry_source':'retained_prior_path'},
])
def test_endpoint_coverage_never_overrides_negative_final_fit_evidence(assessment):
    support=_cylinder(np.linspace(0,5,31))
    traits=compute_traits(_path(np.linspace(0,5,21)),[],support,np.ones(len(support),bool),np.zeros(len(support),int),
                          Normalization(np.zeros(3),1.,1.),primary_centerline_assessment=assessment)
    row=traits.iloc[0]
    assert row.body_coverage_complete
    assert np.isfinite(row.volume)
    assert not row.length_estimate_complete
    assert not row.volume_estimate_complete
    assert not traits.attrs['system_summary']['root_system_volume_estimate_complete']
