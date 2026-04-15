# Transition1x MACE Descriptor Dedup20 Design

## Goal

Create a new Transition1x HDF5 file that keeps at most 20 structures per reaction, selected within each reaction by MACE-OFF23 small hidden-layer descriptor diversity, while preserving the original `data`, `train`, `val`, and `test` layout.

## Input And Model

The input file is `data/Transition1x.h5`. The model file is `~/.mace/MACE-OFF23_small.model`; if it is missing, download it from the MACE-OFF small model URL before processing.

## Selection

For each `data/<formula>/<rxn>` group, extract one descriptor per frame from MACE hidden node features by averaging over atoms. Force-include the frame indices matching `reactant/positions`, `transition_state/positions`, and `product/positions`. Fill the remaining slots up to 20 using farthest-point sampling in descriptor space. If a reaction has 20 or fewer frames, keep all frames.

## Output

Write `data/Transition1x_maceoff_small_dedup20.h5`. For each reaction, copy `atomic_numbers` and the `reactant`, `transition_state`, and `product` subgroups unchanged, slice `positions`, `wB97x_6-31G(d).energy`, and `wB97x_6-31G(d).forces` to the selected frames, and store `selected_frame_indices`. Recreate `train`, `val`, and `test` as hardlinks to the corresponding deduplicated `data` reactions.

## Validation

Run unit tests for endpoint matching and farthest-point selection. Run a small end-to-end smoke test on a limited number of reactions before full processing. Verify the output file opens, has preserved splits, and each reaction has at most 20 frames with endpoint indices present.
