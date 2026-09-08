from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train_multitask_v2 import build_multitask_input


class MultitaskInputsTests(unittest.TestCase):
    def test_rgb_only_does_not_add_fourth_channel(self):
        rgb = torch.zeros(2, 3, 8, 8)
        self.assertIs(build_multitask_input(rgb, None, 3), rgb)

    def test_rgb_nir_without_generator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'loaded generator'):
            build_multitask_input(torch.zeros(2, 3, 8, 8), None, 4)

    def test_nir_is_normalized_and_frozen(self):
        class Generator(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x[:, :1])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        data = build_multitask_input(torch.zeros(2, 3, 8, 8), Generator(), 4, mean, std)
        self.assertEqual(tuple(data.shape), (2, 4, 8, 8))
        self.assertTrue(torch.allclose(data[:, 3], torch.full((2, 8, 8), (0.5-0.485)/0.229)))


if __name__ == '__main__':
    unittest.main()
