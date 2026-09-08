import unittest
import numpy as np
import torch
from src.training.train_meal_official import regression_metrics,losses,should_flip


class MealTrainingTests(unittest.TestCase):
    def test_target_units_and_zero_policy(self):
        m = regression_metrics([[0,100],[200,300]],[[10,110],[180,290]])
        self.assertEqual(m['calories']['mae'],15)
        self.assertEqual(m['mass']['mae'],10)
        self.assertEqual(m['calories']['mape_n'],1)
        self.assertAlmostEqual(m['calories']['mape_nonzero_percent'],10)

    def test_nonfinite_metrics_fail(self):
        with self.assertRaises(ValueError):
            regression_metrics([[1,2]],[[np.nan,2]])

    def test_normalized_loss_gradient(self):
        p = torch.tensor([[20.,50.]],requires_grad=True)
        total,reg = losses(torch.zeros(1,3),p,torch.tensor([[10.,30.]]),torch.tensor([0]),torch.tensor([10.,20.]))
        self.assertEqual(reg.item(),1.)
        total.backward()
        torch.testing.assert_close(p.grad,torch.tensor([[.05,.025]]))

    def test_flip_reproducible(self):
        values = [should_flip('dish_1556572657',e) for e in range(20)]
        self.assertEqual(values,[should_flip('dish_1556572657',e) for e in range(20)])
        self.assertEqual(set(values),{True,False})


if __name__=='__main__':
    unittest.main()
