# Surface continuity during lateral tracing

Mesh inputs use `LocalMeshConnectivity` in lateral growth, tip recovery, and
centerline refinement. Point-cloud-only inputs retain the existing tracing
behavior. Reduced analysis clouds use connectivity from the full-resolution
triangle mesh through a vertex mapping.

## Local surface ownership

Each growth or centering neighbourhood is restricted to the incoming mesh
patch. Connectivity is evaluated in a local ball: a distant connection through
the taproot does not make neighbouring lateral surfaces interchangeable.
Centering anchors the patch to its supplied support vertices, so a nearby
internal cavity cannot change the selected surface merely by being closer to
the centerline.

Nested cavity boundaries require an additional check. Approximately coaxial,
nested cross-section circles may describe the inner and outer boundary of one
tube. These surfaces can support the same trace even when their local mesh
patches are disconnected. A neighbouring tube with a distinct cross-section
center fails that check. Nesting must agree in at least two of three local
sections. A tapered cavity cap can have a disk-like section, so the inner
boundary allows a larger fitting residual while the enclosing exterior must
remain well fit.

Local patches are cached with their anchor, radius, and direction. Identical
effective growth variants reuse a copied trace while retaining the original
candidate multiplicity and parameter identities used by consensus selection.

## Endpoints and fragments

For mesh inputs, endpoint continuation runs before the next tracing order
creates branches. Local surface ownership allows it to recover support that
would otherwise be hidden by provisional point labels. Recovery and centering
alternate for at most three passes; the process stops when no supported
continuation remains.

Before branch selection, `resolve_endpoint_fragments` compares outgoing
fragments jointly against established endpoints. It uses the unsnapped surface
part of each candidate, forward tangent agreement, local connectivity, and an
assignment margin. Existing basal root identities are retained. A selected
fragment's geometry and support are absorbed into its continuation, and its
children are transferred to the established root. Competing assignments are
retained with `endpoint_continuation_ambiguous` for review. Subsequent hierarchy
repair recomputes orders and attachment references.

## Contact and fused surfaces

`TubeSectionTracker` fits one or two boundary-circle modes at successive
cross-sections. A two-mode fit must improve the boundary residual and satisfy
support, radius, and separation checks. This avoids splitting an ordinary
single circular tube into two roots.

Both fitted centers are associated jointly with predicted previous centers.
The incoming root keeps its index as sections move through contact and back
into separation. A temporarily unresolved merged section can retain supported
predictions; these sections produce `fused_surface_identity_ambiguous` QC flags.

This model assumes approximately tubular cross-sections. Severe fusion,
insufficient sampling, highly irregular cross-sections, and contacts lacking
distinct incoming/outgoing evidence still need review. It does not establish
biological parenthood from physical contact alone.

## Validation and diagnostics

`tests/test_surface_tracking.py` covers locally disconnected neighbours with a
distant crown connection, reduced clouds, nested cavities, a short root beside
a longer neighbour, centering with mixed support, fusion/separation sequences,
continuation fragments, real departing branches, and ambiguous assignments.

The result metadata's `lateral_tracing_policy` records enabled surface checks.
Trace score components include step count, step limit, and whether the limit
was reached. Centering records multi-tube and ambiguous section counts in the
in-memory root score components; QC flags survive the standard exports.

Root identifiers are deterministically assigned from the repaired hierarchy.
They can change between complete reruns. Compare biological roots using their
origins and proximal geometry, rather than assuming the same numeric suffix
identifies the same root in two different runs.
