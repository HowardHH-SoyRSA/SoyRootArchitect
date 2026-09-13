"""Interactive 3D result viewer and graph editor.

The editor deliberately lives beside, rather than inside, the automatic
pipeline.  It treats a completed SoyRootBio output directory as an immutable
baseline and materialises edits by replaying an append-only operation log.
"""

from .session import EditorSession, EditorValidationError

__all__ = ["EditorSession", "EditorValidationError"]
