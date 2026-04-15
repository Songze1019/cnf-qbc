# DES370K Raw Data Summary

Source: Zenodo DOI `10.5281/zenodo.5676266`

Local paths:
- archive: `data/des370k/DES370K.zip`
- raw csv: `data/des370k/raw/DES370K.csv`
- metadata: `data/des370k/raw/DES370K_meta.csv`
- geometries: `data/des370k/raw/geometries/`

## Table shape

- rows: `370,959`
- columns: `54`
- unique `system_id`: `3,691`
- unique `geom_id`: `370,959`

## Geometry generation groups

- `md_dimer`: `166,914` (`44.99%`)
- `qm_opt_dimer`: `97,368` (`26.25%`)
- `md_solvation`: `64,476` (`17.38%`)
- `md_nmer`: `42,201` (`11.38%`)

## Charge-pair distribution

- `(0, 0)`: `300,171` (`80.92%`)
- `(0, +1)`: `28,849` (`7.78%`)
- `(0, -1)`: `26,526` (`7.15%`)
- `(+1, 0)`: `10,319` (`2.78%`)
- others combined: `5,094` (`1.37%`)

## Atom-count summary

- total atoms per dimer min / mean / max: `2 / 14.79 / 34`

## `cbs_CCSD(T)_all` interaction energy

- sample count: `370,959`
- min / mean / max: `-169.52 / -1.08 / 99.96 kcal/mol`
- std: `11.01 kcal/mol`
- fraction `< -5 kcal/mol`: `15.65%`
- fraction `< -10 kcal/mol`: `8.07%`
- fraction `> 0 kcal/mol`: `23.61%`

Per-group means:
- `qm_opt_dimer`: `-3.75 kcal/mol`
- `md_solvation`: `-1.11 kcal/mol`
- `md_nmer`: `-0.59 kcal/mol`
- `md_dimer`: `+0.38 kcal/mol`

## Most frequent monomer pairs

- `O` + `O`: `4,869`
- `S` + `S`: `4,850`
- `C` + `C`: `4,805`
- `N` + `N`: `4,712`
- `C=O` + `C=O`: `4,700`
- `O` + `[I-]`: `3,397`
- `O` + `[He]`: `3,327`
- `O` + `[Br-]`: `3,305`
- `O` + `[Cl-]`: `3,270`
- `O` + `S`: `3,221`

## Notes

- The raw CSV and metadata are already available locally and analyzable.
- Geometry extraction was still running when this summary was generated, so the `geometries/` file count may be temporarily lower than the final `geom_id` count until unzip completes.
