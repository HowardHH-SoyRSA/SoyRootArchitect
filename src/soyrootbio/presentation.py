"""Noise-free views of vertices already excluded from scientific measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .io import write_labeled_ply


def geometry_fingerprint(points: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array, dtype in ((points, '<f8'), (triangles, '<i8')):
        value = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def export_noise_free_presentation(
    output_dir: Path, points: np.ndarray, triangles: np.ndarray | None,
    *, colors: np.ndarray, root_ids: np.ndarray, root_orders: np.ndarray,
    assignment_states: np.ndarray, excluded_mask: np.ndarray,
    analysis_noise_mask: np.ndarray | None = None, review: dict | None = None,
) -> dict:
    """Physically omit hidden vertices and remap faces in a presentation-only PLY."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(excluded_mask, dtype=bool)
    if mask.shape != (len(points),):
        raise ValueError('Presentation noise mask must match source vertices')
    faces = np.empty((0, 3), dtype=np.int64) if triangles is None else np.asarray(triangles)
    touched = mask[faces].sum(axis=1)
    if np.any((touched != 0) & (touched != 3)):
        raise ValueError('Presentation exclusion must not cut a connected mesh component')
    core = mask if analysis_noise_mask is None else np.asarray(analysis_noise_mask, dtype=bool)
    if core.shape != mask.shape or not np.array_equal(core, mask):
        raise ValueError('Every hidden noise vertex must also be excluded from analysis')
    if np.any(np.asarray(root_ids)[mask] != -1) or np.any(np.asarray(assignment_states)[mask] != 0):
        raise ValueError('Hidden noise vertices must be unassigned before presentation export')
    indices = np.flatnonzero(~mask)
    if not len(indices):
        raise ValueError('Presentation exclusion cannot remove the whole structure')
    inverse = np.full(len(points), -1, dtype=np.int64)
    inverse[indices] = np.arange(len(indices))
    retained_faces = faces[touched == 0]
    write_labeled_ply(
        output_dir / 'presentation_root_structure.ply', np.asarray(points)[indices],
        triangles=inverse[retained_faces], colors=np.asarray(colors)[indices],
        root_ids=np.asarray(root_ids)[indices], root_orders=np.asarray(root_orders)[indices],
        assignment_states=np.asarray(assignment_states)[indices],
    )
    np.savez_compressed(output_dir / 'presentation_vertex_mapping.npz',
                        presentation_to_source=indices)
    fingerprint = geometry_fingerprint(points, faces)
    np.savez_compressed(output_dir / 'presentation_noise_masks.npz',
                        excluded_full_vertices=mask,
                        source_geometry_sha256=np.array(fingerprint))
    report = {
        'schema': 'soyrootbio.noise-presentation/v2',
        'source_geometry_sha256': fingerprint,
        'mesh_file': 'presentation_root_structure.ply',
        'mapping_file': 'presentation_vertex_mapping.npz',
        'masks_file': 'presentation_noise_masks.npz',
        'source_vertex_count': len(points), 'visible_vertex_count': len(indices),
        'hidden_vertex_count': int(mask.sum()),
        'hidden_face_count': int(np.sum(touched == 3)),
        'analysis_excluded_vertex_count': int(core.sum()),
        'additional_review_hidden_vertex_count': int(np.sum(mask & ~core)),
        'additional_review_changes_analysis_or_traits': False,
        'all_hidden_vertices_excluded_from_assignment_and_traits': True,
        'source_geometry_preserved': True, 'review': review,
    }
    (output_dir / 'presentation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


class NoisePresentation:
    """Validate a display mask and retain native vertex IDs for editor picking."""

    def __init__(self, output_dir: Path, mesh) -> None:
        self.mask = np.zeros(mesh.vertex_count, dtype=bool)
        self.mesh = mesh
        self.cached_path = None
        path = Path(output_dir) / 'presentation_noise_masks.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                mask = np.asarray(data['excluded_full_vertices'], dtype=bool)
                fingerprint = str(data['source_geometry_sha256'].item())
            if fingerprint != geometry_fingerprint(mesh.positions, mesh.triangles):
                raise ValueError('Presentation noise mask belongs to different geometry')
        else:
            path = Path(output_dir) / 'noise_reduction_masks.npz'
            if not path.exists():
                return
            with np.load(path, allow_pickle=False) as data:
                mask = np.asarray(data['excluded_full_vertices'], dtype=bool)
            if mask.shape != self.mask.shape or np.any(mesh.root_labels[mask] != -1):
                raise ValueError('Analysis noise mask is inconsistent with source assignments')
        if mask.shape != self.mask.shape:
            raise ValueError('Presentation noise mask must match source vertices')
        touched = mask[mesh.triangles].sum(axis=1)
        if np.any((touched != 0) & (touched != 3)):
            raise ValueError('Presentation mask would cut a connected source component')
        if np.any(mesh.root_labels[mask] != -1) or np.any(mesh.assignment_states[mask] != 0):
            raise ValueError('Hidden noise vertices still have assignments; regenerate the scientific bundle with the full noise mask')
        self.mask = mask

    def hidden_labels(self) -> set[int]:
        hidden = set(np.unique(self.mesh.root_labels[self.mask]).tolist())
        visible = set(np.unique(self.mesh.root_labels[~self.mask]).tolist())
        return {label for label in hidden - visible if label >= 0}

    def mesh_path(self, session_dir: Path) -> Path:
        if not self.mask.any():
            return self.mesh.path
        if self.cached_path is None:
            # Keep ALL source vertices and IDs; only face visibility changes.
            # The editor renders triangles, so excluded vertices cannot draw or pick.
            path = Path(session_dir) / 'noise_hidden_display.ply'
            faces = self.mesh.triangles[~np.any(self.mask[self.mesh.triangles], axis=1)]
            write_labeled_ply(path, self.mesh.positions, triangles=faces,
                              colors=self.mesh.colors / 255., root_ids=self.mesh.root_labels,
                              root_orders=self.mesh.root_orders,
                              assignment_states=self.mesh.assignment_states)
            self.cached_path = path
        return self.cached_path
