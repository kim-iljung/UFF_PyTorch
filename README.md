# UFF PyTorch

This repository provides a differentiable implementation of the Universal Force
Field (UFF) using PyTorch.  It mirrors the energy expressions from RDKit's
reference implementation while exposing GPU-friendly tensor operations so that
energies and forces can be obtained via autograd.

## Features

- Conversion utilities that map an RDKit `Mol` object with 3D coordinates to
  PyTorch tensors.
- Support for bonded interactions (bond stretch, angle bend, torsions,
  inversions) and non-bonded van der Waals interactions.
- A `torch.nn.Module` wrapper that evaluates the total UFF energy and its
  individual components on CPU or GPU.

## Installation

### Dependencies

The package requires the following Python dependencies (versions shown are the
minimum supported):

- Python 3.8 or newer
- [PyTorch](https://pytorch.org/) 1.12+
- [RDKit](https://www.rdkit.org/) 2022.03.1+
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster) 1.6.0+
- [torch-sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.0+

`torch-cluster` and `torch-sparse` provide the accelerated neighbor search and
sparse linear algebra used by the non-bonded fast path.  Ensure that the wheel
builds for your platform and PyTorch version are available; consult the
respective project documentation if you need to install from source.

You can install the package from a Git checkout once the dependencies are
available in your environment:

```bash
pip install git+https://github.com/kim-iljung/UFF_PyTorch.git
```

## Quick start

```python
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from uff_torch import UFFTorch, build_uff_inputs

mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
AllChem.EmbedMolecule(mol)
inputs = build_uff_inputs(mol, device=torch.device("cuda"))
model = UFFTorch(inputs).to("cuda")
energy = model()  # differentiable total energy
energy.backward()  # compute forces stored in model.reference_coords.grad
```

The helper returns initial coordinates; you can optimise them with any PyTorch
optimizer by treating the coordinate tensor as a learnable parameter.

## Data provenance

The atomic parameter table is extracted from
[`Params.cpp`](https://github.com/rdkit/rdkit/blob/master/Code/ForceField/UFF/Params.cpp)
in the RDKit project (BSD license).  It is stored locally as a JSON file so that
no runtime dependency on RDKit's C++ bindings is required beyond molecule
pre-processing.
