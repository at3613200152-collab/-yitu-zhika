"""Minimal GPU regressions at the actual train_epoch call site."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train_capsicum import train_epoch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA test requires the training GPU')
class AmpTests(unittest.TestCase):
    def test_persistent_overflow_stops_without_changing_weights(self):
        model = torch.nn.Conv2d(3, 1, 1, bias=False).cuda()
        torch.nn.init.constant_(model.weight, 0.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=2.0**30)
        before = model.weight.detach().clone()
        batch = (torch.ones(1, 3, 1, 1), torch.zeros(1, 1, 1, 1))
        with self.assertRaisesRegex(FloatingPointError, 'Persistent AMP overflow'):
            train_epoch(model, [batch]*8, optimizer, scaler, torch.device('cuda'), lambda **v: None)
        self.assertTrue(torch.equal(model.weight, before))

    def test_invalid_data_is_not_hidden_by_amp_recovery(self):
        model = torch.nn.Conv2d(3, 1, 1, bias=False).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cuda')
        batch = (torch.full((1, 3, 1, 1), float('nan')), torch.zeros(1, 1, 1, 1))
        with self.assertRaisesRegex(FloatingPointError, 'input or target'):
            train_epoch(model, [batch], optimizer, scaler, torch.device('cuda'), lambda **v: None)

    def test_finite_loss_amp_overflow_recovers_without_bad_update(self):
        model = torch.nn.Conv2d(3, 1, 1, bias=False).cuda()
        torch.nn.init.constant_(model.weight, 0.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=131072.0)
        rgb = torch.ones(1, 3, 1, 1)
        nir = torch.zeros(1, 1, 1, 1)
        records = []
        loss = train_epoch(model, [(rgb, nir)] * 4, optimizer, scaler,
                           torch.device('cuda'), lambda **v: records.append(v))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertLess(scaler.get_scale(), 131072.0)
        self.assertTrue(torch.all(model.weight < 0.25))
        self.assertTrue(any(v.get('event') == 'amp_overflow' for v in records))


if __name__ == '__main__':
    unittest.main()
