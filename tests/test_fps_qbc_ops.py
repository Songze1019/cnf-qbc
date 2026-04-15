import unittest

import numpy as np

from workflow.utils.fps_qbc_ops import normalize_feature_matrices


class FpsQbcNormalizationTests(unittest.TestCase):
    def test_none_normalization_returns_inputs(self):
        train = np.array([[1.0, 2.0], [3.0, 4.0]])
        pool = np.array([[5.0, 6.0]])

        norm_train, norm_pool = normalize_feature_matrices(train, pool, "none")

        np.testing.assert_allclose(norm_train, train)
        np.testing.assert_allclose(norm_pool, pool)

    def test_zscore_normalization_uses_shared_statistics_and_handles_constant_dims(self):
        train = np.array(
            [
                [0.0, 10.0, 5.0],
                [1.0, 20.0, 5.0],
            ]
        )
        pool = np.array(
            [
                [2.0, 30.0, 5.0],
                [3.0, 40.0, 5.0],
            ]
        )

        norm_train, norm_pool = normalize_feature_matrices(train, pool, "zscore")
        combined = np.concatenate([norm_train, norm_pool], axis=0)

        np.testing.assert_allclose(combined[:, 0].mean(), 0.0, atol=1e-7)
        np.testing.assert_allclose(combined[:, 1].mean(), 0.0, atol=1e-7)
        np.testing.assert_allclose(combined[:, 0].std(), 1.0, atol=1e-7)
        np.testing.assert_allclose(combined[:, 1].std(), 1.0, atol=1e-7)
        np.testing.assert_allclose(combined[:, 2], 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
