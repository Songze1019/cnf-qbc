# Transition1x MACE Scratch Train Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable shell script for training MACE from scratch on the deduplicated Transition1x extxyz splits.

**Architecture:** Keep the implementation to one focused shell script that wraps `mace_run_train` with local defaults and path validation. Reuse the style of the existing local training entrypoint instead of introducing a new Python launcher.

**Tech Stack:** Bash, extxyz, mace_run_train.

---

### Task 1: Add Script

**Files:**
- Create: `scripts/train_transition1x_mace_scratch.sh`

- [ ] Write a bash script with `set -euo pipefail`.
- [ ] Add overridable environment variables for dataset paths, work directories, and common hyperparameters.
- [ ] Validate `mace_run_train` exists and the three xyz files exist.
- [ ] Invoke `mace_run_train` without `--foundation_model`.
- [ ] Allow passthrough extra CLI args via `"$@"`.

### Task 2: Verify Script

**Files:**
- Modify: `scripts/train_transition1x_mace_scratch.sh`

- [ ] Run `bash -n scripts/train_transition1x_mace_scratch.sh`.
- [ ] Make the script executable.
- [ ] Review the rendered command and defaults for consistency with `scripts/train.sh`.
