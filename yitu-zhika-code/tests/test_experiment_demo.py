import unittest
import numpy as np
from app.experiment_pipeline import format_prediction


class DemoContractTests(unittest.TestCase):
    def example(self):
        return dict(category_name='蔬菜', category_prob=.8, rgb_calories=150., rgb_weight=100.,
                    calories=145., weight=98., nir_image=np.zeros((256, 256), dtype=np.uint8),
                    external_calories=None, external_status='训练中')

    def test_pending_external_does_not_invent_zero(self):
        image, category, text = format_prediction(self.example())
        self.assertEqual(image.shape, (256, 256))
        self.assertIn('粗类别', category)
        self.assertIn('训练中', text)
        self.assertIn('不支持', text)

    def test_negative_output_warns_without_clamping(self):
        example = self.example()
        example['calories'] = -12.
        _, _, text = format_prediction(example)
        self.assertIn('-12.0 kcal', text)
        self.assertIn('不合理负值', text)


if __name__ == '__main__':
    unittest.main()
