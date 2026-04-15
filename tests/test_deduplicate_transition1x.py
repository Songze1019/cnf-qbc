import numpy as np
import unittest

from utils.deduplicate_transition1x import (
    farthest_point_indices,
    find_matching_frame_indices,
    select_diverse_indices,
)


class Transition1xDedupTests(unittest.TestCase):
    def test_find_matching_frame_indices_returns_exact_position_matches(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
            ]
        )
        endpoints = [
            positions[0:1],
            positions[2:3],
            positions[0:1],
        ]

        self.assertEqual(find_matching_frame_indices(positions, endpoints), [0, 2])

    def test_farthest_point_indices_keeps_forced_indices_and_adds_distant_points(self):
        descriptors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ]
        )

        selected = farthest_point_indices(
            descriptors, keep_count=4, forced_indices=[0, 4]
        )

        self.assertEqual(len(selected), 4)
        self.assertIn(0, selected)
        self.assertIn(4, selected)
        self.assertIn(2, selected)
        self.assertEqual(selected, sorted(selected))

    def test_select_diverse_indices_keeps_all_when_shorter_than_keep_count(self):
        descriptors = np.arange(12, dtype=float).reshape(6, 2)

        selected = select_diverse_indices(
            descriptors=descriptors,
            endpoint_indices=[0, 3, 5],
            keep_count=20,
        )

        self.assertEqual(selected, [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
