"""PyTorch implementation of the Universal Force Field (UFF).

This package provides utilities to convert an RDKit ``Mol`` object into
tensor representations and a differentiable, GPU-aware ``torch.nn.Module``
that evaluates the bonded and non-bonded energy terms of the Universal
Force Field.  The implementation mirrors the reference RDKit implementation
while exposing vectorised PyTorch kernels so that energies and forces can be
computed with autograd.
"""

from .builder import build_uff_inputs, merge_uff_inputs
from .model import UFFTorch

__version__ = "0.1.0"

__all__ = ["build_uff_inputs", "merge_uff_inputs", "UFFTorch", "__version__"]
