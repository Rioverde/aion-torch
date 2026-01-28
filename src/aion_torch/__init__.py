"""AION: Adaptive Input/Output Normalization for deep neural networks.

AION provides adaptive scaling of residual connections based on energy ratios,
improving training stability in very deep networks.

Basic usage:
    >>> import torch
    >>> from aion_torch import AionResidual
    >>>
    >>> layer = AionResidual(alpha0=0.1, beta=0.05)
    >>> x = torch.randn(8, 512)
    >>> y = torch.randn(8, 512)
    >>> out = layer(x, y)
"""

__version__ = "1.0.0"

from .adapters import AionBlock
from .aion_adapter import AionResidual
from .alpha import compute_alpha
from .energy import energy

__all__ = [
    "AionResidual",
    "AionBlock",
    "energy",
    "compute_alpha",
    "__version__",
]
