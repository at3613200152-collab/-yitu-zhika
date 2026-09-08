import unittest
import torch

from src.models.meal_ablation import NutritionHead, MealAblationNet
from src.training.meal_ablation_core import ablation_loss, select_views, ARMS


class AblationTests(unittest.TestCase):
    def test_consistency_only_compares_distinct_views_of_same_dish(self):
        row = dict(dish_id='dish1', image='a', image_sha256='a',
                   views=[dict(path=k, file_sha256=k) for k in ('a', 'b', 'c')])
        a, b = select_views(row, 'train', 2, 42, paired=True)
        self.assertNotEqual(a['path'], b['path'])
        self.assertEqual((a, b), select_views(row, 'train', 2, 42, paired=True))
        single = dict(dish_id='single', image='s', image_sha256='s')
        a, b = select_views(single, 'train', 2, 42, paired=True)
        self.assertIsNone(b)
        prediction = dict(nutrition=torch.tensor([[100., 50.]], requires_grad=True),
                          logits=torch.zeros(1, 2, requires_grad=True), density=None)
        second = dict(nutrition=torch.tensor([[120., 60.]], requires_grad=True),
                      logits=torch.zeros(1, 2, requires_grad=True), density=None)
        _, details = ablation_loss(prediction, torch.tensor([[100., 50.]]), torch.tensor([0]),
                                  torch.tensor([100., 50.]), 2., second=second,
                                  pair_mask=torch.tensor([True]), consistency_weight=.1)
        self.assertAlmostEqual(details['consistency'].item(), .2, places=6)

    def test_forced_off_gate_removes_nir_influence(self):
        torch.set_num_threads(2)
        manifest = dict(category_to_idx={'a': 0, 'b': 1},
                        target_stats=dict(mean=[200., 100.], std=[100., 50.]),
                        density_stats=dict(mean=2.))
        model = MealAblationNet(manifest, fusion='gate', head='positive', pretrained=False).eval()
        rgb, nir = torch.randn(2, 3, 64, 64), torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            first = model(rgb, nir, gate_override=0.)
            second = model(rgb, nir*7, gate_override=0.)
        torch.testing.assert_close(first['nutrition'], second['nutrition'])
        self.assertTrue((first['gate'] == 0).all())

    def test_factorized_predictions_are_nonnegative_and_reconstruct_calories(self):
        head = NutritionHead('factorized', [200., 100.], [100., 50.], 2.)
        prediction, density = head(torch.tensor([[0., 0.], [-20., -20.]], requires_grad=True))
        torch.testing.assert_close(prediction[0], torch.tensor([200., 100.]))
        torch.testing.assert_close(prediction[:, 0], prediction[:, 1]*density)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())
        prediction.sum().backward()

    def test_clip_auxiliary_requires_explicit_finite_semantics(self):
        manifest = dict(category_to_idx={'a': 0, 'b': 1},
                        target_stats=dict(mean=[200., 100.], std=[100., 50.]), density_stats=dict(mean=2.))
        model = MealAblationNet(manifest, fusion='rgb', pretrained=False, clip_aux=True).eval()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 64, 64))
        result = model(torch.randn(1, 3, 64, 64), clip_scores=torch.tensor([[.2, .1]]))
        result['nutrition'].sum().backward()
        self.assertIsNotNone(model.semantic_adapter.weight.grad)

    def test_zero_mass_does_not_create_fake_density_target(self):
        output = dict(nutrition=torch.tensor([[1., 1.]], requires_grad=True),
                      logits=torch.zeros(1, 2), density=torch.tensor([float('nan')]))
        loss, _ = ablation_loss(output, torch.zeros(1, 2), torch.tensor([0]), torch.ones(2), 2.)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
