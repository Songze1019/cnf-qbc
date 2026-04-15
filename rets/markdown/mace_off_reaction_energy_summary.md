# Transition1x Reaction Energy Evaluation

- Dataset: `data/Transition1x_maceoff_small_dedup20.h5`
- DFT reference: `wB97x/6-31G(d)` endpoint energies from reactant / transition_state / product groups
- Definitions: `forward barrier = E_TS - E_reactant`; `reaction energy = E_product - E_reactant`; `reverse barrier = E_TS - E_product`; error = `model - DFT`
- Unit: `eV`

## Models

- `MACE-OFF small`: `/home/sjtu-caoxiaoming/huosongze/.mace/MACE-OFF23_small.model`; reaction table: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/mace_off_reaction_energies.csv`
- `Scratch lr=5e-3`: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x-mace-alldata/checkpoints/transition1x_scratch_run-0.model`; reaction table: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x_reaction_energies_scratch_lr5e-3.csv`
- `Scratch lr=1e-2`: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x-mace-alldata-lr1e-2/checkpoints/transition1x_scratch_run-0.model`; reaction table: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x_reaction_energies_scratch_lr1e-2.csv`
- `Scratch lr=1e-2 + stage2`: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x-mace-alldata-lr1e-2-stage2/checkpoints/transition1x_scratch_run-0_stagetwo.model`; reaction table: `/home/sjtu-caoxiaoming/huosongze/apps/cnf/rets/transition1x_reaction_energies_scratch_lr1e-2_stage2.csv`

## Overall Comparison

| Model | Fwd MAE | Fwd RMSE | Rxn MAE | Rxn RMSE | Rev MAE | Rev RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MACE-OFF small | 0.551 | 0.745 | 0.511 | 0.765 | 0.633 | 0.817 |
| Scratch lr=5e-3 | 0.088 | 0.156 | 0.096 | 0.128 | 0.101 | 0.170 |
| Scratch lr=1e-2 | 0.096 | 0.167 | 0.099 | 0.134 | 0.105 | 0.174 |
| Scratch lr=1e-2 + stage2 | 0.054 | 0.119 | 0.046 | 0.065 | 0.065 | 0.131 |

## Test Split Comparison

| Model | Fwd MAE | Fwd RMSE | Rxn MAE | Rxn RMSE | Rev MAE | Rev RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MACE-OFF small | 0.523 | 0.723 | 0.568 | 0.871 | 0.642 | 0.865 |
| Scratch lr=5e-3 | 0.105 | 0.151 | 0.102 | 0.135 | 0.113 | 0.167 |
| Scratch lr=1e-2 | 0.117 | 0.170 | 0.104 | 0.143 | 0.124 | 0.177 |
| Scratch lr=1e-2 + stage2 | 0.080 | 0.114 | 0.059 | 0.085 | 0.093 | 0.138 |

## Important Points

- On overall MAE, best forward barrier / reaction energy / reverse barrier are `Scratch lr=1e-2 + stage2` / `Scratch lr=1e-2 + stage2` / `Scratch lr=1e-2 + stage2`.
- On test MAE, best forward barrier / reaction energy / reverse barrier are `Scratch lr=1e-2 + stage2` / `Scratch lr=1e-2 + stage2` / `Scratch lr=1e-2 + stage2`.
- Relative to `MACE-OFF small`, `Scratch lr=1e-2 + stage2` reduces overall MAE from `0.551` to `0.054` for forward barrier, `0.511` to `0.046` for reaction energy, and `0.633` to `0.065` for reverse barrier.
- On the test split, `Scratch lr=1e-2 + stage2` gives `Fwd/Rxn/Rev MAE = 0.080 / 0.059 / 0.093 eV`.

