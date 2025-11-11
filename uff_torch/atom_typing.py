"""Utilities that expose RDKit's UFF atom typing logic."""

from __future__ import annotations

from rdkit.Chem.rdchem import Atom

from ._atom_typing import uff_atom_type as _cxx_uff_atom_type

__all__ = ["uff_atom_type"]


def uff_atom_type(atom: Atom, *, tolerate_charge_mismatch: bool = False) -> str:
    """Return the UFF atom type label used by RDKit."""

    return _cxx_uff_atom_type(atom, tolerate_charge_mismatch)

