"""Independent acceptance examples with known axes, not implementation mirrors."""
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio import centerline
from soyrootbio.geometry import path_length, tangent_vectors


def surface(axis, radius=.08, angles=None):
    """Planar-axis tube with deliberately uneven but complete angular sampling."""
    if angles is None:
        angles = np.sort(np.r_[np.linspace(0, 2*np.pi, 20, endpoint=False),
                               np.linspace(-.35, .35, 17, endpoint=False)] % (2*np.pi))
    points = []
    for station, tangent in zip(axis, tangent_vectors(axis)):
        normal = np.array([0., 0., 1.])
        other = np.cross(tangent, normal)
        points.extend(station + radius * (np.cos(angles)[:, None]*normal
                                         + np.sin(angles)[:, None]*other))
    return np.asarray(points)


def straight(start=0., end=2., count=101):
    return np.column_stack([np.linspace(start, end, count), np.zeros((count, 2))])


@pytest.mark.parametrize('scale', [.01, 1., 100.])
def test_lateral_validation_does_not_accept_a_smooth_but_truncated_prior(scale):
    axis = straight()
    points = surface(axis)*scale
    # All stations in this middle crop are beautifully centred. It still omits
    # most of the observed body and must not win only on local centering score.
    cropped_prior = straight(.6, 1.4, 41)*scale
    line, detail = centerline._fit_lateral_body(points, axis*scale, cropped_prior, .02*scale)
    assert detail['fit_qc_passed']
    assert path_length(line)/scale > 1.85
    assert line[0, 0]/scale < .08
    assert line[-1, 0]/scale > 1.92
    json.dumps(detail, allow_nan=False)


def test_lateral_validation_can_replace_offset_prior_and_overextended_tip():
    axis = straight()
    points = surface(axis)
    prior = straight(0, 2.7, 101) + [0., 0., .055]
    line, detail = centerline._fit_lateral_body(points, axis, prior, .02)
    assert detail['fit_qc_passed']
    assert 1.85 < path_length(line) < 2.1
    assert line[-1, 0] <= 2.02
    assert np.quantile(cKDTree(axis).query(line)[0], .95) < .025


def test_dense_basal_sampling_cannot_hide_a_long_supported_distal_body():
    axis = straight()
    # Dense rings at the base contribute 97% of the vertices. A candidate
    # covering only those rings still omits 40% of the connected root's extent.
    dense = surface(axis[:61], angles=np.linspace(0, 2*np.pi, 200, endpoint=False))
    distal = surface(axis[61:], angles=np.linspace(0, 2*np.pi, 12, endpoint=False))
    points = np.vstack([dense, distal])
    prior = straight(0, 1.2, 61)
    line, detail = centerline._fit_lateral_body(points, axis, prior, .02)
    assert detail['fit_qc_passed']
    assert path_length(line) > 1.85
    assert line[-1, 0] > 1.92


@pytest.mark.parametrize('radius,prior_end', [(.08, 1.7), (.08, 1.9), (.3, 1.8)])
def test_prior_tie_preference_cannot_discard_supported_distal_extent(radius, prior_end):
    axis = straight()
    # Both curves can pass a permissive whole-body coverage floor. The prior
    # must not win the centering tie while a valid full-length curve exists.
    # On thick roots radial envelopes can even give both curves 100% coverage.
    line, detail = centerline._fit_lateral_body(
        surface(axis, radius=radius), axis, straight(0., prior_end, 86), .02)
    assert detail['fit_qc_passed']
    assert path_length(line) > 1.92
    assert line[-1, 0] > 1.92


def test_lateral_validation_preserves_a_real_large_bend():
    angle = np.linspace(0, np.pi, 161)
    axis = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(angle))])
    points = surface(axis, radius=.06)
    line, detail = centerline._fit_lateral_body(points, axis, axis.copy(), .02)
    assert detail['fit_qc_passed']
    assert 3.0 < path_length(line) < 3.3
    assert np.quantile(cKDTree(axis).query(line)[0], .95) < .03
    assert path_length(line)/np.linalg.norm(line[-1]-line[0]) > 1.45


def test_prior_rejection_is_not_converted_to_measurement_acceptance_on_a_wall():
    axis = straight()
    angles = np.linspace(-.4*np.pi, .4*np.pi, 31)
    points = surface(axis, angles=angles)
    # A mesh-derived mean axis can sit inside the convex hull of a wall even
    # when the actual tubular cross-section is not surrounded by observations.
    pilot = axis + [0., 0., .05]
    line, detail = centerline._fit_lateral_body(points, pilot, axis.copy(), .01)
    assert not detail['fit_qc_passed']
    assert not detail['fit_applied']
    assert detail['traits_geometry_source'] == 'retained_prior'
    np.testing.assert_array_equal(line, axis)
    json.dumps(detail, allow_nan=False)
