import unittest
import torch
from src.models.generator import UNetGenerator
from src.training.train_meal_official import MealNet,seed_epoch
from src.training.train_meal_nir_official import NirMealNet


class NirMealTests(unittest.TestCase):
    def test_shared_initialization_and_frozen_generator(self):
        torch.set_num_threads(2)
        m={'category_to_idx':{'a':0,'b':1},'target_stats':{'mean':[250.,200.],'std':[200.,150.]}}
        generator=UNetGenerator(base_filters=64)
        seed_epoch(0)
        rgb=MealNet(m,pretrained=False)
        seed_epoch(0)
        nir=NirMealNet(m,generator.state_dict(),pretrained=False)
        for key,value in rgb.network.state_dict().items():
            other=nir.network.state_dict()[key]
            if key=='features.0.weight':
                torch.testing.assert_close(value,other[:,:3])
                self.assertEqual(torch.count_nonzero(other[:,3]).item(),0)
            else:
                torch.testing.assert_close(value,other)
        nir.train()
        self.assertFalse(nir.generator.training)
        self.assertTrue(all(not p.requires_grad for p in nir.generator.parameters()))
        nir.eval()
        with torch.no_grad():
            logits,values=nir(torch.zeros(1,3,256,256))
        self.assertEqual(values.shape,(1,2))
        self.assertTrue(torch.isfinite(values).all())


if __name__=='__main__':
    unittest.main()
