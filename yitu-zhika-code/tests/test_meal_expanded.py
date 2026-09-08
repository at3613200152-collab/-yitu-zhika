import copy
import unittest

from src.data.meal_expanded_manifest import merge_training_dishes, validate_expansion_labels


class ExpandedManifestTests(unittest.TestCase):
    def test_rejects_unapproved_ids_and_mismatched_official_targets(self):
        extra = {'rows': [dict(dish_id='b', split='train', targets=[300, 150],
                              capture_day_utc='2019-02-28')]}
        with self.assertRaises(ValueError):
            validate_expansion_labels(extra, {'a'}, {'b'}, {'b': [300, 150]})
        with self.assertRaises(ValueError):
            validate_expansion_labels(extra, {'b'}, set(), {'b': [150, 300]})

    def test_adds_dishes_only_to_training_and_preserves_holdouts(self):
        base = {'target_names': ['calories', 'mass'], 'category_to_idx': {'other': 0},
                'rows': [dict(dish_id=k, split=s, targets=t, category='other', category_idx=0,
                              image=k+'.png', image_sha256=k, pixel_sha256=k)
                         for k, s, t in [('a', 'train', [100, 50]), ('v', 'val', [999, 900]),
                                         ('t', 'test', [888, 800])]]}
        extra = {'rows': [dict(dish_id='b', split='train', category='other', targets=[300, 150],
                              frames=[dict(path='b.png', file_sha256='b', pixel_sha256='b')])]}
        before = copy.deepcopy(base)
        result = merge_training_dishes(base, extra)
        self.assertEqual(result['counts'], {'train': 2, 'val': 1, 'test': 1})
        self.assertEqual(result['target_stats']['mean'], [200., 100.])
        self.assertEqual([r for r in result['rows'] if r['split'] != 'train'], base['rows'][1:])
        self.assertEqual(base, before)


if __name__ == '__main__':
    unittest.main()
