# SoyRoot Studio

Local, full-resolution 3D inspection and non-destructive graph editing for
SoyRootBio result bundles.

The browser renders the labelled PLY through WebGL with the high-performance
GPU preference. PLY parsing, normal calculation, and the spatial picking index
run in workers. Vertex count is preserved; the UI does not downsample the mesh.

## Build

Requirements: Node.js 22.13 or newer and pnpm 11.

```powershell
pnpm install
pnpm run typecheck
pnpm run lint
pnpm run build
```

The build retains the Vinext output and also writes a static application to
`dist/client/index.html`. The SoyRootBio Python editor server detects and serves
that directory directly:

```powershell
python -m soyrootbio.editor.server --output "path\to\SoyRootBio_outputs\sample"
```

The interface normally uses the same origin as the Python API. For development
on a separate frontend port, append
`?api=http://127.0.0.1:8765` to the frontend URL.

## Editing model

- Automatic PLY, hierarchy, and traits files remain read-only.
- Create-from-unassigned-path, split, merge, assign, reconnect, reparent,
  delete, redraw, and root-order correction are sent to the local operation
  API.
- Undo and redo are persisted as append-only history events.
- Export materialises edited PLY, JSON, CSV, RSML, and the operation log into
  the editor session directory.
- A single hierarchy click highlights and frames the complete selected root,
  its direct parent, direct children, insertion point, and tip point.
- The same left panel lists every triangle-connected uncertain and unassigned
  point patch for direct selection, highlighting, framing, and inspection.
- Successful edits and history actions preserve the active camera orbit, zoom,
  and target while retaining the selected-root highlight.
- The default full-system orbit pivots around the geometric centre of the
  primary root.
- Shift + left-dragging in Assign mode paints one continuous 3D brush region
  and records the entire stroke as one undoable operation; ordinary left-drag
  remains available to rotate.
- New roots are drawn base-to-tip through grey unassigned points, attached to
  the selected parent, and claim only nearby unassigned vertices.

The renderer uses the backend-provided float64 render origin before converting
positions to float32 GPU buffers. Clicked coordinates are converted back to the
original coordinate system before an edit is submitted.
