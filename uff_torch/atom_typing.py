"""Utilities that reproduce RDKit's UFF atom typing logic in Python."""

from __future__ import annotations

import warnings

from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import Atom, HybridizationType


_PT = GetPeriodicTable()


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _has_conjugated_bond(atom: Atom) -> bool:
    for bond in atom.GetBonds():
        if bond.GetIsConjugated():
            return True
    return False


def _append_charge(
    atom: Atom,
    atom_key: str,
    *,
    tolerate_charge_mismatch: bool = False,
) -> str:
    total_valence = atom.GetTotalValence()
    formal_charge = atom.GetFormalCharge()
    atomic_num = atom.GetAtomicNum()

    def matches(expected: int) -> bool:
        return (
            total_valence == expected
            or formal_charge == expected
            or tolerate_charge_mismatch
        )

    def ensure(expected: int) -> None:
        nonlocal atom_key
        if matches(expected):
            atom_key += f"+{expected}"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )

    if atomic_num in {29, 47}:  # Cu, Ag
        ensure(1)
    elif atomic_num in {4, 20, 25, 26, 28, 46, 78}:  # Be, Ca, Mn, Fe, Ni, Pd, Pt
        ensure(2)
    elif atomic_num in {21, 24, 27, 79, 89, 96, 97, 98, 99, 100, 101, 102, 103}:
        ensure(3)
    elif atomic_num in {2, 18, 22, 36, 54, 90, 91, 92, 93, 94, 95}:
        ensure(4)
    elif atomic_num in {23, 41, 43, 73}:
        ensure(5)
    elif atomic_num == 42:
        ensure(6)
    elif atomic_num == 12:  # Mg
        if total_valence == 2:
            atom_key += "+2"
        else:
            if tolerate_charge_mismatch:
                atom_key += "+2"
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 13:  # Al
        if total_valence != 3 and not tolerate_charge_mismatch:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 14:  # Si
        if total_valence != 4 and not tolerate_charge_mismatch:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 15:  # P
        if total_valence == 3:
            atom_key += "+3"
        elif total_valence == 5 or tolerate_charge_mismatch:
            atom_key += "+5"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 16:  # S
        if atom.GetHybridization() != HybridizationType.SP2:
            if total_valence == 2:
                atom_key += "+2"
            elif total_valence == 4:
                atom_key += "+4"
            elif total_valence == 6 or tolerate_charge_mismatch:
                atom_key += "+6"
            else:
                _warn(
                    f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
                )
    elif atomic_num == 30:  # Zn
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 31:  # Ga
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 33:  # As
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 34:  # Se
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 48:  # Cd
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 49:  # In
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 51:  # Sb
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 52:  # Te
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 75:  # Re
        if tolerate_charge_mismatch:
            if atom_key == "Re6":
                atom_key = "Re6+5"
            elif atom_key == "Re3":
                atom_key = "Re3+7"
        _warn(
            f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
        )
    elif atomic_num == 80:  # Hg
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 81:  # Tl
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 82:  # Pb
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 83:  # Bi
        if total_valence == 3 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )
    elif atomic_num == 84:  # Po
        if total_valence == 2 or tolerate_charge_mismatch:
            atom_key += "+2"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )

    if 57 <= atomic_num <= 71:  # lanthanides
        if total_valence == 6 or tolerate_charge_mismatch:
            atom_key += "+3"
        else:
            _warn(
                f"UFF atom typing: unrecognized charge state for atom {atom.GetIdx()}"
            )

    return atom_key


def uff_atom_type(atom: Atom, *, tolerate_charge_mismatch: bool = False) -> str:
    """Return the UFF atom type label used by RDKit."""

    symbol = atom.GetSymbol()
    atom_key = f"{symbol}{'_' if len(symbol) == 1 else ''}"
    atomic_num = atom.GetAtomicNum()

    if atomic_num:
        default_valence = _PT.GetDefaultValence(atomic_num)
        outer_electrons = _PT.GetNOuterElecs(atomic_num)
        if default_valence == -1 or (outer_electrons not in (1, 7)):
            if atomic_num in {12, 13, 14, 15, 50, 51, 52, 81, 82, 83, 84}:
                atom_key += "3"
                if atom.GetHybridization() != HybridizationType.SP3:
                    _warn(
                        f"UFF atom typing: forcing SP3 hybridization for atom {atom.GetIdx()}"
                    )
            elif atomic_num == 80:
                atom_key += "1"
                if atom.GetHybridization() != HybridizationType.SP:
                    _warn(
                        f"UFF atom typing: forcing SP hybridization for atom {atom.GetIdx()}"
                    )
            else:
                hyb = atom.GetHybridization()
                if hyb == HybridizationType.S:
                    pass
                elif hyb == HybridizationType.SP:
                    atom_key += "1"
                elif hyb == HybridizationType.SP2:
                    if (
                        (atom.GetIsAromatic() or _has_conjugated_bond(atom))
                        and atomic_num in {6, 7, 8, 16}
                    ):
                        atom_key += "R"
                    else:
                        atom_key += "2"
                elif hyb == HybridizationType.SP3:
                    atom_key += "3"
                elif hyb == HybridizationType.SP2D:
                    atom_key += "4"
                elif hyb == HybridizationType.SP3D:
                    atom_key += "5"
                elif hyb == HybridizationType.SP3D2:
                    atom_key += "6"
                else:
                    _warn(
                        f"UFF atom typing: unrecognized hybridization for atom {atom.GetIdx()}"
                    )

    atom_key = _append_charge(
        atom, atom_key, tolerate_charge_mismatch=tolerate_charge_mismatch
    )
    return atom_key
