"""Real CUDA tests are opt-in; CPU tests also work without CuPy installed."""
import os
from contextvars import copy_context

import numpy as np
import pytest

from soyrootbio.gpu_backend import active_backend, compute_backend
from soyrootbio.pipeline import _nearest_exposed_segments, _ExposedSegmentIndexCache


def test_backend_selection_is_explicit_and_context_local():
    with compute_backend("cpu") as backend:
        assert backend is None and active_backend() is None
        assert copy_context().run(active_backend) is None
    with pytest.raises(ValueError, match="cpu or cuda"):
        with compute_backend("auto"):
            pass


@pytest.mark.skipif(os.environ.get("SOYROOTBIO_TEST_CUDA") != "1", reason="requires CUDA opt-in")
@pytest.mark.parametrize("scale", [1e-7, 1.0, 1e7])
def test_cuda_exact_distinct_root_competition_and_bounded_batches(scale):
    rng = np.random.default_rng(218)
    points = rng.uniform(-0.1, 0.1, (4100, 3)) * scale
    points[:4] = [[0, 0, 0], [0, scale*.01, 0], [scale*.1, 0, 0], [scale*10, 0, 0]]
    paths = [
        (8, np.array([[-.1, 0, 0], [0, 0, 0], [.1, 0, 0]]) * scale),
        (2, np.array([[-.1, 0, 0], [.1, 0, 0]]) * scale),  # exact distinct-root tie
        (11, np.array([[0, .02, 0], [0, .02, 0]]) * scale),  # zero length
        (19, rng.uniform(-.1, .1, (12, 3)) * scale),
    ]
    args = dict(d_bar=.01*scale, radius=.07*scale, margin=.01*scale)
    with compute_backend("cpu"):
        expected = _nearest_exposed_segments(points, paths, **args)
    with compute_backend("cuda") as backend:
        backend.max_pairs = 113  # exercise multiple transfer/kernel chunks
        cache = _ExposedSegmentIndexCache()
        actual = _nearest_exposed_segments(points, paths, segment_index_cache=cache, **args)
        launches = backend.stats["kernel_launches"]
        again = _nearest_exposed_segments(points, paths, segment_index_cache=cache, **args)
        assert backend.stats["kernel_launches"] == launches > 1
        assert backend.stats["sparse_pairs"] >= backend.stats["cpu_refined_pairs"] > 0
    for a, b, c in zip(expected, actual, again):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    assert active_backend() is None


@pytest.mark.skipif(os.environ.get("SOYROOTBIO_TEST_CUDA") != "1", reason="requires CUDA opt-in")
def test_cuda_empty_single_root_and_relabelled_index():
    args = dict(d_bar=.01, radius=.02, margin=.001)
    cache = _ExposedSegmentIndexCache()
    with compute_backend("cuda"):
        result = _nearest_exposed_segments(np.empty((0, 3)), [], **args)
        assert all(len(x) == 0 for x in result)
        for label in (1, 17):
            result = _nearest_exposed_segments(np.array([[0., 0., 0.]]),
                [(label, np.array([[0., 0., 0.]]))], segment_index_cache=cache, **args)
            assert result[0][0] == 0 and result[1][0] == label
            assert np.isinf(result[2][0]) and result[3][0] == -1
