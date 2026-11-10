"""Native cross-sections and disconnected pieces cannot fabricate axial volume."""
import numpy as np
import pytest

from soyrootbio.measurement_evidence import build_measurement_evidence
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization


def _native_tube(length, radius, *, axis=(0., 0., 1.), origin=(0., 0., 0.),
                 rings=3, sides=12):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(sides) * 2 * np.pi / sides
    centers = np.asarray(origin) + np.linspace(0., length, rings)[:, None] * axis
    surface = centers[:, None] + radius * (
        np.cos(angles)[None, :, None] * u + np.sin(angles)[None, :, None] * v)
    points = np.vstack([surface.reshape(-1, 3), centers[[0, -1]]])
    faces = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    for j in range(sides):
        faces.append([rings * sides, (j + 1) % sides, j])
        faces.append([rings * sides + 1, (rings - 1) * sides + j,
                      (rings - 1) * sides + (j + 1) % sides])
    return points, np.asarray(faces), centers


def _measure(points, faces, path, *, spacing):
    labels = np.zeros(len(points), int)
    evidence = build_measurement_evidence(
        points, labels, 0, path, triangles=faces, spacing=spacing,
        identity_resolved=False, attachment_supported=False, tip_supported=False)
    assessment = {"status": "fitted", "fit_qc_passed": True,
                  "identity_qc_passed": False, "measurement_evidence": evidence}
    traits = compute_traits(
        path, [], points, labels == 0, labels, Normalization(np.zeros(3), 1., 1.),
        full_points=points, triangles=faces, full_root_labels=labels,
        primary_centerline_assessment=assessment)
    return traits.iloc[0], evidence


def test_single_native_cross_section_does_not_create_axial_length_or_frustum():
    theta = np.arange(20) * 2 * np.pi / 20
    points = np.vstack([np.zeros((1, 3)),
                        np.c_[.1 * np.cos(theta), .1 * np.sin(theta), np.zeros(20)]])
    faces = np.array([[0, i + 1, (i + 1) % 20 + 1] for i in range(20)])
    path = np.array([[0., 0., -.03], [0., 0., .03]])
    row, evidence = _measure(points, faces, path, spacing=.03)
    assert np.ptp(points[:, 2]) == 0.
    assert evidence["supported_interval_mask"] == [False]
    assert np.isnan(row.length) and np.isnan(row.volume)
    assert row.length_measurement_status == row.volume_measurement_status == "unavailable"
    assert row.surface_area > 0.  # Preserve the actually observed native disk.


@pytest.mark.parametrize("length,radius,rings,sides,spacing", [
    (.01, .1, 3, 12, .03),  # Thinner than the sampling tolerance, but observed.
    (.5, 1., 2, 6, .1),   # Coarse valid longitudinal native edges.
])
def test_actual_thin_and_coarse_native_tubes_keep_positive_measured_length(
        length, radius, rings, sides, spacing):
    points, faces, centers = _native_tube(length, radius, rings=rings, sides=sides)
    row, evidence = _measure(points, faces, centers[[0, -1]], spacing=spacing)
    assert evidence["supported_interval_mask"] == [True]
    assert row.length == pytest.approx(length)
    assert row.volume == pytest.approx(np.pi * radius ** 2 * length, rel=1e-6)


def _separated_3d_tubes():
    axis = np.array([1., 2., 2.]) / 3.
    first, first_faces, first_centers = _native_tube(.06, .01, axis=axis)
    second, second_faces, second_centers = _native_tube(
        .02, .01, axis=axis, origin=.08 * axis)
    return (np.vstack([first, second]),
            np.vstack([first_faces, second_faces + len(first)]),
            first_centers, second_centers)


def test_coarse_interval_cannot_bridge_known_native_gap_even_with_half_overlap():
    points, faces, first, second = _separated_3d_tubes()
    # One observed fragment covers 60% of this interval. Endpoint tolerance
    # alone would incorrectly fill the missing interval between the pieces.
    path = np.vstack([first[0], second[-1]])
    row, evidence = _measure(points, faces, path, spacing=.05)
    assert evidence["supported_interval_mask"] == [False]
    assert np.isnan(row.length) and np.isnan(row.volume)


def test_separate_observed_3d_intervals_measure_without_gap_volume():
    points, faces, first, second = _separated_3d_tubes()
    row, evidence = _measure(points, faces, np.vstack([first, second]), spacing=.05)
    assert evidence["supported_interval_mask"] == [True, True, False, True, True]
    assert row.length == pytest.approx(.08)
    assert row.volume == pytest.approx(np.pi * .01 ** 2 * .08, rel=1e-6)
    assert np.isnan(row.tortuosity)
    assert row.length_measurement_status == row.volume_measurement_status == "partial"
