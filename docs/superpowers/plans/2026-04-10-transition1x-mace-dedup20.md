# Transition1x MACE Descriptor Dedup20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a Transition1x deduplication pipeline that keeps 20 MACE-diverse structures per reaction while preserving the HDF5 split layout.

**Architecture:** Add one focused script for Transition1x HDF5 processing and descriptor-based selection. Keep pure selection helpers testable without loading MACE; isolate model loading and HDF5 copying in CLI functions.

**Tech Stack:** Python, h5py, numpy, torch, ase, mace, e3nn, pytest.

---

### Task 1: Selection Helpers

**Files:**
- Create: `utils/deduplicate_transition1x.py`
- Create: `tests/test_deduplicate_transition1x.py`

- [ ] Write tests for endpoint frame matching and farthest-point sampling with forced indices.
- [ ] Run `pytest tests/test_deduplicate_transition1x.py -q` and verify the tests fail because the module is missing.
- [ ] Implement `find_matching_frame_indices`, `farthest_point_indices`, and `select_diverse_indices`.
- [ ] Run `pytest tests/test_deduplicate_transition1x.py -q` and verify the tests pass.

### Task 2: HDF5 And Descriptor Pipeline

**Files:**
- Modify: `utils/deduplicate_transition1x.py`

- [ ] Add CLI args for input/output/model/device/batch-size/keep-count/max-reactions/overwrite.
- [ ] Add MACE-OFF23 small download to `~/.mace/MACE-OFF23_small.model`.
- [ ] Add per-reaction ASE Atoms construction from Transition1x frames.
- [ ] Add batched MACE descriptor extraction using the pattern from `utils/deduplicate.py`.
- [ ] Add HDF5 output writing with sliced frame datasets, copied endpoint groups, `selected_frame_indices`, attrs, and split hardlinks.
- [ ] Add a smoke-run mode with `--max-reactions`.

### Task 3: Validation And Full Run

**Files:**
- Generated: `data/Transition1x_maceoff_small_dedup20.h5`

- [ ] Run unit tests.
- [ ] Download or verify the MACE-OFF23 small model in `~/.mace`.
- [ ] Run a smoke test with `--max-reactions 2`.
- [ ] Run the full pipeline if the smoke test succeeds.
- [ ] Verify the full output opens, split hardlinks point to `data`, endpoints are retained, and every reaction has no more than 20 frames.
