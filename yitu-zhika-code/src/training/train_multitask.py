"""Compatibility entry point replacing the obsolete two-target trainer.

This entry now delegates to the explicitly five-target v2 trainer. It remains
experimental: official grouped train/validation/test benchmarking is separate.
"""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_multitask_v2 import main, train_multitask_v2


def train_multitask(config, resume=False, generator_ckpt=None):
    warnings.warn('Legacy entry now uses verified labels and the 5-target v2 trainer.', stacklevel=2)
    return train_multitask_v2(config, resume=resume, generator_ckpt=generator_ckpt)


if __name__ == '__main__':
    main()
