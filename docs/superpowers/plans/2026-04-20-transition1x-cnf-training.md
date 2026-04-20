# Transition1x CNF Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable single-xyz CNF training entrypoint for transition1x that uses a minimal shared runner instead of another fully duplicated training script.

**Architecture:** Create one small shared runner module that owns the repeated single-xyz training flow: data loading, `AtomicData` conversion, alias setup, dataloader creation, model construction, callbacks, and `Trainer.fit()`. Add a transition1x-specific CLI script that only defines defaults and forwards parsed arguments into the shared runner.

**Tech Stack:** Python, Lightning, PyTorch, PyG via `mace.tools.torch_geometric`, ASE/extxyz, unittest.

---

### Task 1: Add failing tests for the new training CLI surface

**Files:**
- Create: `tests/test_train_transition1x_cnf.py`
- Test: `tests/test_train_transition1x_cnf.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path

from src.train_transition1x_cnf import build_parser, default_out_root


class TrainTransition1xCliTests(unittest.TestCase):
    def test_parser_defaults_target_transition1x_single_xyz_training(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.data_path, Path("data/transition1x/train.xyz"))
        self.assertEqual(args.out_root, Path("src/transition1x/cnf"))
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.max_epochs, 1000)
        self.assertFalse(args.pbc)

    def test_default_out_root_matches_transition1x_training_layout(self) -> None:
        self.assertEqual(default_out_root(), Path("src/transition1x/cnf"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbols for `src.train_transition1x_cnf`.

- [ ] **Step 3: Write minimal implementation**

```python
def default_out_root() -> Path:
    return Path("src/transition1x/cnf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--data-path", type=Path, default=Path("data/transition1x/train.xyz"))
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    ...
    parser.add_argument("--pbc", action="store_true", default=False)
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_transition1x_cnf.py src/train_transition1x_cnf.py
git commit -m "feat: add transition1x cnf training cli"
```

### Task 2: Add failing tests for the shared runner path helpers

**Files:**
- Create: `src/trainers/__init__.py`
- Create: `src/trainers/cnf_runner.py`
- Modify: `tests/test_train_transition1x_cnf.py`
- Test: `tests/test_train_transition1x_cnf.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
...
from src.trainers.cnf_runner import build_run_paths


class CnfRunnerPathTests(unittest.TestCase):
    def test_build_run_paths_creates_expected_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, ckpt_dir = build_run_paths(Path(tmpdir), "20260420-120000")

            self.assertEqual(run_dir, Path(tmpdir) / "20260420-120000")
            self.assertEqual(ckpt_dir, run_dir / "checkpoints")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: FAIL with missing module or missing `build_run_paths`.

- [ ] **Step 3: Write minimal implementation**

```python
def build_run_paths(out_root: Path, timestamp: str) -> tuple[Path, Path]:
    run_dir = out_root / timestamp
    ckpt_dir = run_dir / "checkpoints"
    return run_dir, ckpt_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_transition1x_cnf.py src/trainers/__init__.py src/trainers/cnf_runner.py
git commit -m "feat: add shared cnf runner helpers"
```

### Task 3: Implement the shared single-xyz training flow

**Files:**
- Modify: `src/trainers/cnf_runner.py`
- Modify: `src/train_transition1x_cnf.py`
- Test: `tests/test_train_transition1x_cnf.py`

- [ ] **Step 1: Write the failing test**

```python
from src.train_transition1x_cnf import make_runner_config


class TrainTransition1xConfigTests(unittest.TestCase):
    def test_make_runner_config_sets_non_pbc_transition1x_defaults(self) -> None:
        args = build_parser().parse_args([])
        config = make_runner_config(args)

        self.assertEqual(config["data_path"], Path("data/transition1x/train.xyz"))
        self.assertEqual(config["out_root"], Path("src/transition1x/cnf"))
        self.assertFalse(config["pbc"])
        self.assertEqual(config["num_elements"], 83)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: FAIL with missing `make_runner_config`.

- [ ] **Step 3: Write minimal implementation**

```python
class EpochPrintCallback(Callback):
    ...


def run_single_xyz_training(config: dict) -> Path:
    ...
    configurations = load_from_xyz(str(config["data_path"]))
    atomic_datas = [AtomicData.from_config(cfg, cutoff=config["cutoff"]) for cfg in configurations]
    for data in atomic_datas:
        setattr(data, "atomic_numbers", data.node_attrs)
        setattr(data, "pos", data.positions)
    ...
    trainer.fit(model, train_dataloaders=train_loader)
    return run_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_transition1x_cnf.py src/trainers/cnf_runner.py src/train_transition1x_cnf.py
git commit -m "feat: add single-xyz cnf training runner"
```

### Task 4: Verify script startup on transition1x data

**Files:**
- Modify: `src/train_transition1x_cnf.py`
- Modify: `src/trainers/cnf_runner.py`
- Test: `tests/test_train_transition1x_cnf.py`

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest discover -s tests -p 'test_train_transition1x_cnf.py' -v`
Expected: PASS

- [ ] **Step 2: Run a minimal startup command**

Run: `python src/train_transition1x_cnf.py --max-epochs 1 --batch-size 2 --limit-configs 4`
Expected: The script prints the configuration, creates `src/transition1x/cnf/<timestamp>/checkpoints`, and starts a one-epoch Lightning run.

- [ ] **Step 3: Inspect output directories**

Run: `find src/transition1x/cnf -maxdepth 2 -type d | sort | tail -n 5`
Expected: Includes the new timestamped run directory and its `checkpoints` subdirectory.

- [ ] **Step 4: Commit**

```bash
git add src/train_transition1x_cnf.py src/trainers/cnf_runner.py tests/test_train_transition1x_cnf.py docs/superpowers/plans/2026-04-20-transition1x-cnf-training.md
git commit -m "feat: add transition1x cnf training entrypoint"
```
