from .train_nir_generator import train_one_epoch as train_nir_one_epoch
from .train_multitask import train_one_epoch as train_multitask_one_epoch

__all__ = [
    "train_nir_one_epoch",
    "train_multitask_one_epoch",
]
