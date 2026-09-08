import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.verify_nutrition5k_metadata import audit, legacy_checkpoint_metadata
from src.models.checkpoint_io import named_nutrition, resolve_metadata


class MetadataTests(unittest.TestCase):
    def test_unknown_checkpoint_cannot_receive_legacy_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'
            path.write_bytes(b'future model')
            with self.assertRaisesRegex(ValueError, 'historical checkpoint'):
                legacy_checkpoint_metadata(path)

    def test_missing_model_does_not_produce_zero_calorie_estimate(self):
        from app.pipeline import InferencePipeline
        from PIL import Image
        pipe = InferencePipeline(device='cpu')
        with self.assertRaisesRegex(ValueError, 'loaded multitask'):
            pipe.predict(Image.new('RGB', (32, 32)))

    def test_proven_swapped_columns(self):
        truth = {'dish_x': {'total_calories':300.794281,'total_mass':193,
                           'total_fat':12,'total_carb':28,'total_protein':18}}
        row = {'dish_id':'dish_x', **truth['dish_x']}
        row['total_calories'],row['total_mass'] = row['total_mass'],row['total_calories']
        result = audit([row], truth)
        self.assertEqual(result['swapped_matches'], 1)
        self.assertEqual(result['as_named_matches'], 0)

    def test_legacy_output_names_restore_physical_units(self):
        self.assertEqual(named_nutrition([193,300.794281], ['mass','calories']),
                         {'mass':193.0, 'calories':300.794281})

    def test_unknown_two_head_model_not_assumed_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'target_names'):
                resolve_metadata({}, Path(d)/'unknown.pt', 2)

    def test_wrong_sidecar_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pt'
            path.write_bytes(b'new model')
            Path(str(path)+'.metadata.json').write_text(json.dumps({
                'checkpoint_sha256':hashlib.sha256(b'old model').hexdigest(),
                'target_names':['mass','calories']}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'sidecar hash'):
                resolve_metadata({}, path, 2)


if __name__ == '__main__':
    unittest.main()
