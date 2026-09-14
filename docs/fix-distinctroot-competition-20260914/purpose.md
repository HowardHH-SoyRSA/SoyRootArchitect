# Purpose

The previous assignment logic queried the two nearest skeleton nodes. Both nodes could belong to one densely sampled root, hiding a nearby segment from another root and suppressing valid ambiguity evidence.

This change makes competition root-aware and segment-aware. It computes the closest projection for each relevant root, compares the two closest distinct root identities, and adjusts centerline distance by an interpolated local radius when support is available. A bounded segment index keeps full-resolution assignment practical and deterministic for sparse paths, endpoints, repeated points, zero-length segments, and numerical ties.

The same distinct-root evidence now drives uncertain labels, parent-owned junction decisions, collar reporting, primary patch cleanup, and primary-to-child ownership correction. The exported evidence and metadata make downstream decisions auditable.

The validation report records both end-to-end output changes and comparisons with tracing held fixed. The observed reductions in retained roots and changes in traits are material and require biological review before the branch is merged.
