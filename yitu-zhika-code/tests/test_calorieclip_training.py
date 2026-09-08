import unittest
import numpy as np
from src.training.train_calorieclip_official import scalar_metrics


class CalorieMetricsTests(unittest.TestCase):
    def test_zero_labels_kept_for_absolute_errors(self):
        result = scalar_metrics([0, 100], [-10, 90])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['mae'], 10)
        self.assertEqual(result['mape_n'], 1)
        self.assertEqual(result['negative_predictions'], 1)
        self.assertAlmostEqual(result['mape_nonzero_percent'], 10)

    def test_all_zero_targets_have_no_percentage_or_r2(self):
        result = scalar_metrics([0, 0], [1, 2])
        self.assertIsNone(result['r2'])
        self.assertIsNone(result['mape_nonzero_percent'])

    def test_bad_arrays_rejected(self):
        for y, p in [([], []), ([1], [1, 2]), ([1], [np.nan]), ([[1]], [[2]])]:
            with self.assertRaises(ValueError):
                scalar_metrics(y, p)


if __name__ == '__main__':
    unittest.main()
