# Project rules for future changes

These rules apply to all changes in this repository, including tracing, segmentation, topology, attachment, cleanup, editing, validation, and export.

1. **Keep the primary-root top immutable.** Once a primary-root top is selected, preserve that same biological reference throughout tracing, topology, attachment, and validation. A shortened final centerline fit must not move or redefine it.
2. **Preserve collar and origin boundaries.** Always leave points above the collar unassigned. Lateral-root origins at every order must remain below the immutable primary-root top.
3. **Prohibit higher-order contact with the primary.** An order-2 or higher lateral root must not directly contact the primary root. Check native mesh contact when mesh data is available. Resolve supported segmentation errors without inventing geometry; report any contact that cannot be safely resolved as an unresolved violation, not as a compliant result.
