# Point-only evidence benchmark

Assigned synthetic fixtures isolate point evidence. This is not segmentation accuracy, real-soybean validation, or a guarantee of physical contact.

| Fixture | Points | Native reference contact | Proximity observed | Tangent observed | Seconds |
| --- | ---: | --- | --- | --- | ---: |
| connected_tube_seam | 1920 | True | True | True | 0.0094 |
| separated_tubes | 1920 | False | False | False | 0.0090 |
| small_unmeshed_gap | 1920 | False | True | True | 0.0090 |
| near_parallel_sheets | 1152 | None | True | False | 0.0066 |
| unequal_density | 1440 | None | True | False | 0.0068 |
| missing_sectors_at_contact | 1664 | None | False | False | 0.0078 |
| collar_excluded_child | 1920 | True | False | False | 0.0085 |
| occupied_cylinder | 3915 | None | True | False | 0.0199 |
| sparse_points | 2 | None | False | False | 0.0007 |
| scaling_640 | 640 | True | False | False | 0.0031 |
| scaling_3200 | 3200 | True | True | True | 0.0165 |
| scaling_12800 | 12800 | True | False | False | 0.0593 |

All cases retain unresolved native contact/patch status, make zero ownership changes, and generate no mesh.

A small physical gap can yield both proximity and tangent evidence. Missing contact samples can hide real contact. These cases explain why diagnostic neighborhoods cannot certify the repository's native-mesh rules.

The JSON records full reports, geometry hashes, fixed seeds and runtime versions. Synthetic timings depend on the machine and sampling density; no real-data accuracy claim follows.
