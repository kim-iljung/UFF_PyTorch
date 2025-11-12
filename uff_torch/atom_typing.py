# atom_typing.py
# Utilities that reproduce (and robustify) RDKit's UFF atom typing in Python.
# - BCNO, P, S, H: hybridization + resonance-aware rules
# - Metals / post-transition / lanthanides / actinides: table-driven defaults
#   with a few heuristics (coordination-count) for elements that have multiple UFF types.
#
# References for UFF naming and defaults:
#   - Rappé et al., JACS 1992 (UFF), and commonly used type list conventions like:
#     Zn3+2, Se3+2, Fe3+2 / Fe6+2, Pd4+2, Ag1+1, Hg1+2, etc.

from __future__ import annotations
import warnings
from typing import Dict, Optional, Iterable

from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import Atom, BondType, HybridizationType

_PT = GetPeriodicTable()

# ----------------------------- utilities -----------------------------

def _warn(msg: str, *, tolerate: bool) -> None:
    if not tolerate:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

def _neighbors(atom: Atom) -> Iterable[Atom]:
    # rdchem.Atom.GetNeighbors() returns neighbor Atom objects
    return atom.GetNeighbors()

def _heavy_coordination_number(atom: Atom) -> int:
    # Count neighbors excluding hydrogens (H=1). Include dative/coord bonds as neighbors.
    return sum(1 for nbr in _neighbors(atom) if nbr.GetAtomicNum() != 1)

def _count_double_bonded_oxygens(atom: Atom) -> int:
    cnt = 0
    for b in atom.GetBonds():
        if b.GetBondType() == BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8:
                cnt += 1
    return cnt

def _has_conjugated_bond(atom: Atom) -> bool:
    for b in atom.GetBonds():
        if b.GetIsConjugated():
            return True
    return False

def _base_with_underscore(symbol: str) -> str:
    # UFF naming uses an underscore for 1-letter element symbols (e.g., C_3, N_R, V_3+5)
    return f"{symbol}{'_' if len(symbol) == 1 else ''}"

# ------------------------ organic/nonmetal rules ---------------------

def _type_for_boron(atom: Atom) -> str:
    # B: sp2 (resonant or not) -> B_2, sp3 -> B_3
    base = _base_with_underscore("B")
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP3:
        return base + "3"
    elif hyb == HybridizationType.SP2:
        return base + "2"
    elif hyb == HybridizationType.SP:
        # Very rare; default to trigonal if SP detected without planarity evidence
        return base + "2"
    else:
        # Fallback: degree <= 3 -> assume trigonal (B_2), else sp3
        return base + ("2" if atom.GetDegree() <= 3 else "3")

def _rc_suffix_for_sp2(atom: Atom) -> str:
    # "R" = resonant (aromatic/conjugated) vs "2" = simple trigonal
    if atom.GetIsAromatic() or _has_conjugated_bond(atom):
        return "R"
    return "2"

def _type_for_carbon(atom: Atom) -> str:
    base = _base_with_underscore("C")
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP:
        return base + "1"
    if hyb == HybridizationType.SP2:
        return base + _rc_suffix_for_sp2(atom)
    if hyb == HybridizationType.SP3:
        return base + "3"
    # Rare hypercoord cases: square planar/trig bipyramidal/octahedral
    if hyb == HybridizationType.SP2D:
        return base + "4"
    if hyb == HybridizationType.SP3D:
        return base + "5"
    if hyb == HybridizationType.SP3D2:
        return base + "6"
    # Unknown: default via degree
    deg = atom.GetDegree()
    if deg <= 2:
        return base + "1"
    if deg == 3:
        return base + _rc_suffix_for_sp2(atom)
    return base + "3"

def _type_for_nitrogen(atom: Atom) -> str:
    base = _base_with_underscore("N")
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP:
        return base + "1"
    if hyb == HybridizationType.SP2:
        return base + _rc_suffix_for_sp2(atom)
    if hyb == HybridizationType.SP3:
        return base + "3"
    if hyb == HybridizationType.SP2D:
        return base + "4"
    if hyb == HybridizationType.SP3D:
        return base + "5"
    if hyb == HybridizationType.SP3D2:
        return base + "6"
    # Default by degree
    deg = atom.GetDegree()
    if deg <= 2:
        return base + "1"
    if deg == 3:
        return base + _rc_suffix_for_sp2(atom)
    return base + "3"

def _type_for_oxygen(atom: Atom) -> str:
    base = _base_with_underscore("O")
    hyb = atom.GetHybridization()
    if atom.GetIsAromatic():
        return base + "R"
    if hyb == HybridizationType.SP:
        return base + "1"
    if hyb == HybridizationType.SP2:
        return base + _rc_suffix_for_sp2(atom)
    if hyb == HybridizationType.SP3:
        return base + "3"
    # Default: rely on degree
    deg = atom.GetDegree()
    if deg == 1:
        # terminal O often sp2 (carbonyl) or sp3 (alcohol); prefer sp2 if conjugated
        return base + (_rc_suffix_for_sp2(atom) if _has_conjugated_bond(atom) else "3")
    if deg == 2:
        return base + _rc_suffix_for_sp2(atom)
    return base + "3"

def _type_for_phosphorus(atom: Atom, *, tolerate: bool) -> str:
    # UFF uses tetrahedral P_3+3 / P_3+5 (and a rare P_3+q)
    base = _base_with_underscore("P")
    # Hypervalency heuristic: any P=O or high valence -> +5
    n_pdouble_o = _count_double_bonded_oxygens(atom)
    total_valence = atom.GetTotalValence()
    charge = "+5" if (n_pdouble_o >= 1 or total_valence >= 5) else "+3"
    hyb = atom.GetHybridization()
    if hyb != HybridizationType.SP3:
        _warn(
            f"UFF atom typing (P): forcing SP3 label per UFF ({base}3{charge}); RDKit hyb={hyb}",
            tolerate=tolerate,
        )
    return base + "3" + charge

def _type_for_sulfur(atom: Atom) -> str:
    # S: S_R (aromatic), S_2 (sp2 non-aromatic), S_3+2/4/6 (sp3, hypervalent by # of S=O)
    base = _base_with_underscore("S")
    if atom.GetIsAromatic():
        return base + "R"
    hyb = atom.GetHybridization()
    if hyb == HybridizationType.SP2:
        return base + "2"
    # sp3 or unspecified -> charge tier by # of S=O
    n_sdouble_o = _count_double_bonded_oxygens(atom)
    if n_sdouble_o >= 2:
        return base + "3+6"
    if n_sdouble_o == 1:
        return base + "3+4"
    return base + "3+2"

def _type_for_hydrogen(atom: Atom) -> str:
    # Bridge H between two B: H_b, otherwise H_
    if atom.GetDegree() == 2:
        nbrs = list(_neighbors(atom))
        if len(nbrs) == 2 and nbrs[0].GetAtomicNum() == 5 and nbrs[1].GetAtomicNum() == 5:
            return "H_b"
    return "H_"

# ----------------------- fixed/default element types -----------------

# Elements whose UFF type does not depend (much) on local hybridization.
# Strings are EXACT UFF labels (with underscores where required).
_FIXED_UFF_TYPES: Dict[int, str] = {
    # noble gases
    2: "He4+4", 10: "Ne4+4", 18: "Ar4+4", 36: "Kr4+4", 54: "Xe4+4", 86: "Rn4+4",
    # halogens
    9: "F_", 17: "Cl", 35: "Br", 53: "I_", 85: "At",
    # alkali
    3: "Li", 11: "Na", 19: "K_", 37: "Rb", 55: "Cs", 87: "Fr",
    # alkaline earth
    4: "Be3+2", 12: "Mg3+2", 20: "Ca6+2", 38: "Sr6+2", 56: "Ba6+2", 88: "Ra6+2",
    # p-block (post-1st row) defaults
    13: "Al3", 14: "Si3", 31: "Ga3+3", 32: "Ge3", 49: "In3+3", 50: "Sn3",
    51: "Sb3+3", 52: "Te3+2", 81: "Tl3+3", 82: "Pb3", 83: "Bi3+3", 84: "Po3+2",
    33: "As3+3", 34: "Se3+2",
    # coinage / group 12
    29: "Cu3+1", 47: "Ag1+1", 79: "Au4+3",
    30: "Zn3+2", 48: "Cd3+2", 80: "Hg1+2",
    # early/late transition (single default)
    21: "Sc3+3", 22: "Ti3+4", 23: "V_3+5", 24: "Cr6+3", 25: "Mn6+2",
    27: "Co6+3", 28: "Ni4+2", 45: "Rh6+3", 46: "Pd4+2",
    44: "Ru6+2",
    39: "Y_3+3", 40: "Zr3+4", 41: "Nb3+5",
    72: "Hf3+4", 73: "Ta3+5",
    76: "Os6+6", 77: "Ir6+3", 78: "Pt4+2",
    # lanthanides
    57: "La3+3",
    58: "Ce6+3", 59: "Pr6+3", 60: "Nd6+3", 61: "Pm6+3", 62: "Sm6+3", 63: "Eu6+3",
    64: "Gd6+3", 65: "Tb6+3", 66: "Dy6+3", 67: "Ho6+3", 68: "Er6+3", 69: "Tm6+3",
    70: "Yb6+3", 71: "Lu6+3",
    # actinides
    89: "Ac6+3", 90: "Th6+4", 91: "Pa6+4", 92: "U_6+4", 93: "Np6+4", 94: "Pu6+4",
    95: "Am6+4", 96: "Cm6+3", 97: "Bk6+3", 98: "Cf6+3", 99: "Es6+3", 100: "Fm6+3",
    101: "Md6+3", 102: "No6+3", 103: "Lr6+3",
}

# Elements with multiple UFF variants: pick by (approx.) coordination.
# We keep the rule simple and conservative.
def _type_for_iron(atom: Atom) -> str:
    # Fe: tetra-ish (3) vs octa (6). Use heavy neighbor count >=6 as octahedral.
    cn = _heavy_coordination_number(atom)
    return "Fe6+2" if cn >= 6 else "Fe3+2"

def _type_for_molybdenum(atom: Atom) -> str:
    # Mo: Mo6+6 (octa) vs Mo3+6 (tetra). Use >=6 neighbors as octahedral.
    cn = _heavy_coordination_number(atom)
    return "Mo6+6" if cn >= 6 else "Mo3+6"

def _type_for_tungsten(atom: Atom) -> str:
    # W: W_6+6 (octa) vs W_3+6 (tetra) and seldom W_3+4.
    # Without reliable oxidation state detection, prefer +6 and choose geometry by CN.
    cn = _heavy_coordination_number(atom)
    return "W_6+6" if cn >= 6 else "W_3+6"

def _type_for_rhenium(atom: Atom) -> str:
    # Re: Re6+5 (octa) vs Re3+7 (tetra). Choose by CN.
    cn = _heavy_coordination_number(atom)
    return "Re6+5" if cn >= 6 else "Re3+7"

_MULTI_VARIANT_METALS: Dict[int, callable] = {
    26: _type_for_iron,
    42: _type_for_molybdenum,
    74: _type_for_tungsten,
    75: _type_for_rhenium,
}

# ---------------------------- public API -----------------------------

def uff_atom_type(atom: Atom, *, tolerate_charge_mismatch: bool = True) -> str:
    """
    Return the UFF atom type string expected by RDKit/standard UFF parameter tables.

    Strategy:
    - H, B, C, N, O, P, S: hybridization/resonance rules (R/1/2/3/4/5/6 + P/S charges)
    - Metals/others: table-driven defaults (Zn3+2, Pd4+2, Ag1+1, Hg1+2, ...).
      For Fe/Mo/W/Re select geometry by neighbor count.

    This routine is intentionally conservative for metals: it does NOT try to infer
    oxidation state from RDKit valence (which is often unreliable for dative bonds).
    """
    z = atom.GetAtomicNum()
    if z == 0:
        _warn("UFF atom typing: atomic number 0 is not supported.", tolerate=tolerate_charge_mismatch)
        return "C_3"  # benign fallback

    # First: organic / common nonmetals
    if z == 1:
        return _type_for_hydrogen(atom)
    if z == 5:
        return _type_for_boron(atom)
    if z == 6:
        return _type_for_carbon(atom)
    if z == 7:
        return _type_for_nitrogen(atom)
    if z == 8:
        return _type_for_oxygen(atom)
    if z == 15:
        return _type_for_phosphorus(atom, tolerate=tolerate_charge_mismatch)
    if z == 16:
        return _type_for_sulfur(atom)

    # Second: fixed/default types
    if z in _FIXED_UFF_TYPES:
        return _FIXED_UFF_TYPES[z]

    # Third: metals with multiple UFF variants (geometry by coordination)
    if z in _MULTI_VARIANT_METALS:
        return _MULTI_VARIANT_METALS[z](atom)

    # If we fell through (very unlikely): craft a reasonable default.
    # Use element symbol + default "3" geometry if p-block, otherwise try octahedral 6.
    symbol = atom.GetSymbol()
    base = _base_with_underscore(symbol)

    # Simple family-based fallback using periodic table metadata
    outer = _PT.GetNOuterElecs(z)
    period = _PT.GetRow(z)

    if symbol in {"Se", "Te", "Po"}:
        return base + "3+2"
    if symbol in {"As", "Sb", "Bi"}:
        return base + "3+3"
    if symbol in {"Ge", "Sn", "Pb"}:
        return base + "3"
    if symbol in {"Ga", "In", "Tl"}:
        return base + "3+3"

    # Last-resort metal guess
    cn = _heavy_coordination_number(atom)
    if cn >= 6:
        return base + "6+2"  # many divalent metals are parameterized as 6+2
    if cn == 4:
        return base + "4+2"  # square-planar-ish default
    return base + "3+2"      # tetra +2 as a gentle default

def uff_types_for_mol(mol, *, tolerate_charge_mismatch: bool = True) -> Dict[int, str]:
    """Convenience: return {atom_idx: UFF_type} for the whole molecule."""
    out = {}
    for a in mol.GetAtoms():
        out[a.GetIdx()] = uff_atom_type(a, tolerate_charge_mismatch=tolerate_charge_mismatch)
    return out
    
