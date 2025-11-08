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

You can install the package from a Git checkout:

```bash
pip install git+https://github.com/openai/UFF_PyTorch.git
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
