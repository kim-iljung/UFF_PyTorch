"""Atomic parameters for the Universal Force Field.

The parameter table is sourced from the open-source RDKit project and
converted into JSON so that it can be consumed without relying on the C++
implementation at runtime.  Each entry corresponds to a UFF atom type such
as ``"C_3"`` or ``"O_3"`` and stores the constants required by the
analytical potential functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import math
from typing import Dict, Iterable


@dataclass(frozen=True)
class UFFAtomParameters:
    """Container for the constants associated with a UFF atom type."""

    label: str
    r1: float
    theta0: float  # stored in radians
    x1: float
    D1: float
    zeta: float
    Z1: float
    V1: float
    U1: float
    gmp_xi: float
    gmp_hardness: float
    gmp_radius: float


@lru_cache(maxsize=1)
def _load_table() -> Dict[str, UFFAtomParameters]:
    data_path = resources.files("uff_torch.data").joinpath("uff_atomic_params.json")
    payload = json.loads(data_path.read_text())
    table: Dict[str, UFFAtomParameters] = {}
    for entry in payload:
        table[entry["label"]] = UFFAtomParameters(
            label=entry["label"],
            r1=float(entry["r1"]),
            theta0=math.radians(float(entry["theta0"])),
            x1=float(entry["x1"]),
            D1=float(entry["D1"]),
            zeta=float(entry["zeta"]),
            Z1=float(entry["Z1"]),
            V1=float(entry["V1"]),
            U1=float(entry["U1"]),
            gmp_xi=float(entry["GMP_Xi"]),
            gmp_hardness=float(entry["GMP_Hardness"]),
            gmp_radius=float(entry["GMP_Radius"]),
        )
    return table


def get_atom_parameters(label: str) -> UFFAtomParameters:
    """Return the parameters for a given UFF atom type.

    Parameters
    ----------
    label:
        The UFF atom type label, for example ``"C_3"`` or ``"O_2"``.
    """

    table = _load_table()
    try:
        return table[label]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Unknown UFF atom type '{label}'.") from exc


def list_atom_types() -> Iterable[str]:
    """Return an iterable with the known UFF atom type labels."""

    return _load_table().keys()
