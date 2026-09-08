from .nir_generator import UNetGenerator, Pix2PixModel, PatchGANDiscriminator
from .multitask_net import MultiTaskResNet

# Import from baseline.py module directly (avoid name conflict with baseline/ package)
import importlib.util as _ilu
import os as _os
_baseline_py = _os.path.join(_os.path.dirname(__file__), "baseline.py")
if _os.path.exists(_baseline_py):
    _spec = _ilu.spec_from_file_location("models_baseline_mod", _baseline_py)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaselineResNet = _mod.BaselineResNet
    CalorieCLIPBaseline = _mod.CalorieCLIPBaseline
else:
    BaselineResNet = None
    CalorieCLIPBaseline = None

__all__ = [
    "UNetGenerator",
    "Pix2PixModel",
    "PatchGANDiscriminator",
    "MultiTaskResNet",
    "BaselineResNet",
    "CalorieCLIPBaseline",
]
