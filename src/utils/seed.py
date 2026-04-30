# Utilities for setting random seeds.

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set basic Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)

