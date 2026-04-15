from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflow.utils import fps_only_ops


class TestFpsOnlyOps(unittest.TestCase):
    def test_select_all_local_indices_returns_arange(self) -> None:
        indices = fps_only_ops.select_all_local_indices(5)
        np.testing.assert_array_equal(indices, np.arange(5, dtype=np.int64))

    def test_cmd_select_all_candidates_writes_local_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            candidate_pool_indices = tmpdir_path / "candidate_pool_indices.npy"
            selected_local_npy = tmpdir_path / "selected_local.npy"

            np.save(candidate_pool_indices, np.array([4, 1, 7, 9], dtype=np.int64))

            args = fps_only_ops.build_parser().parse_args(
                [
                    "select-all-candidates",
                    "--candidate_pool_indices_npy",
                    str(candidate_pool_indices),
                    "--selected_local_out",
                    str(selected_local_npy),
                ]
            )
            args.func(args)

            saved = np.load(selected_local_npy)
            np.testing.assert_array_equal(saved, np.arange(4, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
