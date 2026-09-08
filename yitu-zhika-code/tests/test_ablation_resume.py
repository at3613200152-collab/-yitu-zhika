from pathlib import Path
import tempfile
import unittest
import torch

from src.training.train_meal_ablation import resume_state


class ResumeTests(unittest.TestCase):
    def test_resume_rejects_premature_completion(self):
        config = {'epochs': 30, 'patience': 8}
        state = dict(config=config, epoch=1, best_epoch=1, training_complete=True,
                     history=[dict(epoch=1, val=dict(reg_normalized_l1=.5))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                resume_state(path, config)


if __name__ == '__main__':
    unittest.main()
