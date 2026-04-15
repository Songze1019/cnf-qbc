from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from workflow.utils import qbc_ops


class TestQbcOps(unittest.TestCase):
    def test_sample_candidate_indices_is_deterministic_and_unique(self) -> None:
        indices_a = qbc_ops.sample_candidate_indices(20, 5, seed=123)
        indices_b = qbc_ops.sample_candidate_indices(20, 5, seed=123)
        indices_c = qbc_ops.sample_candidate_indices(20, 5, seed=124)

        self.assertEqual(indices_a.tolist(), indices_b.tolist())
        self.assertEqual(len(indices_a), 5)
        self.assertEqual(len(np.unique(indices_a)), 5)
        self.assertTrue(np.all(indices_a >= 0))
        self.assertTrue(np.all(indices_a < 20))
        self.assertNotEqual(indices_a.tolist(), indices_c.tolist())

    def test_cmd_random_subset_writes_candidate_xyz_and_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pool_xyz = tmpdir_path / "pool.xyz"
            candidate_xyz = tmpdir_path / "candidate.xyz"
            candidate_indices = tmpdir_path / "candidate_indices.npy"

            frames = []
            for idx in range(10):
                atoms = Atoms("H", positions=[[float(idx), 0.0, 0.0]])
                atoms.info["frame_id"] = idx
                atoms.info["REF_energy"] = float(idx)
                atoms.arrays["REF_forces"] = np.zeros((1, 3), dtype=float)
                frames.append(atoms)
            write(pool_xyz, frames, format="extxyz")

            args = qbc_ops.build_parser().parse_args(
                [
                    "random-subset",
                    "--pool_xyz",
                    str(pool_xyz),
                    "--candidate_k",
                    "4",
                    "--seed",
                    "7",
                    "--candidate_xyz",
                    str(candidate_xyz),
                    "--candidate_pool_indices",
                    str(candidate_indices),
                ]
            )
            args.func(args)

            saved_indices = np.load(candidate_indices)
            self.assertEqual(saved_indices.shape, (4,))
            self.assertEqual(len(np.unique(saved_indices)), 4)

            candidate_frames = read(candidate_xyz, index=":")
            self.assertEqual(len(candidate_frames), 4)
            frame_ids = [int(atoms.info["frame_id"]) for atoms in candidate_frames]
            self.assertEqual(frame_ids, saved_indices.tolist())


if __name__ == "__main__":
    unittest.main()
