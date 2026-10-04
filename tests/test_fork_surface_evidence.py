import numpy as np
from scipy.spatial import cKDTree

from soyrootbio.fork_evidence import native_fork_surface_evidence
from soyrootbio.types import RootPath


def fixture(short_radius=3., long_radius=3., offcenter=False):
    spacing = .001
    incoming = np.column_stack((np.arange(81), np.zeros(81), np.zeros(81))) * spacing
    short = np.column_stack((80 + .5*np.arange(1, 25), .866*np.arange(1, 25), np.zeros(24))) * spacing
    long = np.column_stack((80 + .8*np.arange(101), -.6*np.arange(101), np.zeros(101))) * spacing
    angle = np.arange(24)*2*np.pi/24
    def tube(path, radius):
        tangent = path[-1]-path[0];tangent /= np.linalg.norm(tangent)
        first = np.cross(tangent, [0., 0., 1.])
        radius = np.broadcast_to(radius, len(path))*spacing
        return np.array([p+r*(np.cos(a)*first+np.sin(a)*np.array([0., 0., 1.]))
                         for p,r in zip(path,radius) for a in angle])
    parent_support = np.vstack((tube(incoming, 3.), tube(short, short_radius)))
    child_support = tube(long, long_radius)
    cloud = np.vstack((parent_support, child_support))
    path = np.vstack((incoming, short))
    if offcenter:
        path[81:, 2] += 4*spacing
    parent = RootPath('parent', path, order=1, covered_indices=set(range(len(parent_support))),
                      score_components={'trace_local_radius': 3*spacing})
    child = RootPath('child', long, order=2, parent_id='parent', insertion_index=80,
                     covered_indices=set(range(len(parent_support),len(cloud))),
                     score_components={'trace_local_radius': 3*spacing})
    return parent, child, cloud, spacing


def evidence(parent, child, cloud, spacing, unsafe=None):
    if unsafe is None:
        unsafe = np.zeros(len(cloud),bool)
    return native_fork_surface_evidence(parent,child,80,cloud,spacing,cKDTree(cloud),unsafe)


def test_equal_caliber_does_not_favor_an_arm_for_being_longer():
    p,c,cloud,d = fixture()
    result = evidence(p,c,cloud,d)
    assert result is not None
    assert abs(result['surface_evidence_gain']) < .03


def test_closed_taper_is_distinguished_from_a_continuing_tube():
    p,c,cloud,d = fixture(short_radius=np.linspace(3.,1.,24))
    result = evidence(p,c,cloud,d)
    assert result['surface_evidence_gain'] > .2
    assert result['closed_cap_evidence'] == 1
    assert result['long_arm_radius_similarity'] > .9


def test_surface_trace_is_weaker_than_a_centered_tube():
    p,c,cloud,d = fixture(offcenter=True)
    result = evidence(p,c,cloud,d)
    assert result['short_arm_centered_section_fraction'] < .5
    assert result['surface_evidence_gain'] > .4


def test_open_or_excluded_support_cannot_authorize_a_change():
    p,c,cloud,d = fixture(short_radius=1.)
    unsafe = np.zeros(len(cloud), bool)
    unsafe[list(c.covered_indices)] = True
    assert evidence(p,c,cloud,d,unsafe) is None


def test_incomplete_incoming_surface_cannot_authorize_a_change():
    p,c,cloud,d = fixture(short_radius=1.)
    p.covered_indices = {i for i in p.covered_indices if cloud[i,2] > d}
    assert evidence(p,c,cloud,d) is None


def test_missing_short_arm_ownership_is_not_surface_trace_evidence():
    p,c,cloud,d = fixture()
    p.covered_indices = {i for i in p.covered_indices if cloud[i,0] < .080 or cloud[i,2] > d}
    assert evidence(p,c,cloud,d) is None


def test_native_surface_evidence_is_independent_of_support_order():
    p,c,cloud,d = fixture(short_radius=1.)
    expected = evidence(p,c,cloud,d)
    order = np.random.default_rng(10).permutation(len(cloud))
    inverse = np.argsort(order)
    p.covered_indices = set(inverse[list(p.covered_indices)])
    c.covered_indices = set(inverse[list(c.covered_indices)])
    actual = evidence(p,c,cloud[order],d)
    assert actual == expected
