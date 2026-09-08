from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.meal_manifest import assign_split, target_stats, training_partition


class ManifestTests(unittest.TestCase):
    def test_official_test_never_becomes_training(self):
        self.assertEqual(assign_split('dish_1563566626', set(), {'dish_1563566626'}), 'test')

    def test_overlapping_official_splits_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assign_split('dish_1563566626', {'dish_1563566626'}, {'dish_1563566626'})

    def test_nearby_same_day_captures_share_validation_assignment(self):
        self.assertEqual(training_partition('dish_1563566626'), training_partition('dish_1563566726'))

    def test_target_statistics_do_not_consume_holdouts(self):
        rows = [{'split':'train', 'targets':[10,20]}, {'split':'train','targets':[30,40]},
                {'split':'test','targets':[10000,10000]}, {'split':'val','targets':[9000,9000]}]
        self.assertEqual(target_stats(rows)['mean'], [20,30])


if __name__ == '__main__':
    unittest.main()
