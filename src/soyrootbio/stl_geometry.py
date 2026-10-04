"""Conservative exact-coordinate indexing of STL facet records.

STL has facets, not shared vertex indices. Decode-time normal splits must not
multiply density evidence. Only closed, oriented manifold vertex fans may be
stitched; open, overlapping and ambiguous fans retain their decoded indices.
"""
from __future__ import annotations

import numpy as np


def index_stl_facets(points: np.ndarray, triangles: np.ndarray, o3d):
    unique, first, inverse = np.unique(
        points, axis=0, return_index=True, return_inverse=True,
    )
    candidate_faces = inverse[triangles]
    unsafe = np.zeros(len(unique), dtype=bool)
    # Nonmanifold vertices include bow ties (two otherwise closed fans touching
    # at one point), which edge counts alone cannot detect.
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(unique),
        o3d.utility.Vector3iVector(candidate_faces),
    )
    unsafe[np.asarray(mesh.get_non_manifold_vertices(), dtype=int)] = True
    del mesh
    edges = np.vstack([candidate_faces[:, [0, 1]], candidate_faces[:, [1, 2]],
                       candidate_faces[:, [2, 0]]])
    canonical_edges, edge_inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True,
    )
    winding = np.bincount(edge_inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    bad_edges = (counts != 2) | (winding != 0)
    unsafe[canonical_edges[bad_edges].ravel()] = True
    # Repeated facets, including opposite-facing coincident sheets, must never
    # provide the apparent two-sided evidence required to close a seam.
    faces, face_counts = np.unique(np.sort(candidate_faces, axis=1), axis=0, return_counts=True)
    unsafe[faces[face_counts > 1].ravel()] = True
    degenerate = np.any(np.diff(np.sort(candidate_faces, axis=1), axis=1) == 0, axis=1)
    degenerate |= np.linalg.norm(np.cross(
        unique[candidate_faces[:, 1]] - unique[candidate_faces[:, 0]],
        unique[candidate_faces[:, 2]] - unique[candidate_faces[:, 0]],
    ), axis=1) == 0
    unsafe[candidate_faces[degenerate].ravel()] = True

    representative = first[inverse]
    ambiguous_source = unsafe[inverse]
    representative[ambiguous_source] = np.flatnonzero(ambiguous_source)
    # Keep first-occurrence source order, matching indexed mesh imports.
    full_to_source, source_to_full = np.unique(representative, return_inverse=True)
    full_points = points[full_to_source]
    full_faces = source_to_full[triangles]
    # Even unresolved seams contribute only one observation per position to
    # tracing. Their full facet connectivity is retained separately for QC.
    analysis_pool = np.sort(source_to_full[first])
    unresolved = np.flatnonzero(unsafe[inverse[full_to_source]])
    report = {
        "policy": "stl-exact-closed-manifold-fans-v1",
        "status": "unresolved_stl_connectivity" if len(unresolved) else "supported_exact_seams",
        "decoded_vertex_count": int(len(points)),
        "distinct_position_count": int(len(unique)),
        "merged_vertex_record_count": int(len(points) - len(full_points)),
        "unresolved_position_count": int(unsafe.sum()),
        "unresolved_full_vertex_count": int(len(unresolved)),
        "analysis_duplicate_records_excluded": int(len(full_points) - len(analysis_pool)),
        "coordinate_tolerance": 0.0,
        "faces_added_or_removed": 0,
        "mapping_file": "input_geometry_mapping.npz",
    }
    return full_points, full_faces, analysis_pool, source_to_full, full_to_source, unresolved, report
