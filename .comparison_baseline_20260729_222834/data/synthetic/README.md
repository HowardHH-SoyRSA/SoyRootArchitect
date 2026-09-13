# Synthetic Test Dataset

`synthetic_taproot.csv` is a small generated point cloud with one primary taproot and three lateral branches.

`synthetic_taproot_endpoints.csv` contains the primary-root endpoints used by the end-to-end tests and README examples.

Regenerate it with:

```powershell
soyrootbio generate-synthetic --output data/synthetic/synthetic_taproot.csv --lateral-count 3 --seed 21
```
