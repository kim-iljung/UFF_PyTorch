"""Mathematical helpers used by the UFF PyTorch implementation."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

from .parameters import UFFAtomParameters

# Physical constants taken from the RDKit implementation
UFF_LAMBDA = 0.1332
UFF_G = 332.06
_GROUP6 = {8, 16, 34, 52, 84}


def clamp_cos(value: float) -> float:
    """Clamp cosine values to the valid range [-1, 1]."""

    return max(-1.0, min(1.0, value))


def calc_bond_rest_length(bond_order: float, end1: UFFAtomParameters,
                          end2: UFFAtomParameters) -> float:
    """Return the UFF equilibrium bond length for a pair of atoms."""

    bond_order = max(bond_order, 1.0e-3)
    ri = end1.r1
    rj = end2.r1
    r_bo = -UFF_LAMBDA * (ri + rj) * math.log(bond_order)
    xi = end1.gmp_xi
    xj = end2.gmp_xi
    sqrt_xi = math.sqrt(xi)
    sqrt_xj = math.sqrt(xj)
    r_en = ri * rj * (sqrt_xi - sqrt_xj) ** 2 / max(xi * ri + xj * rj, 1.0e-8)
    return ri + rj + r_bo - r_en


def calc_bond_force_constant(rest_length: float, end1: UFFAtomParameters,
                              end2: UFFAtomParameters) -> float:
    """Return the harmonic bond force constant."""

    return 2.0 * UFF_G * end1.Z1 * end2.Z1 / max(rest_length**3, 1.0e-12)


def calc_angle_force_constant(theta0: float, bond_order12: float, bond_order23: float,
                              atom1: UFFAtomParameters, atom2: UFFAtomParameters,
                              atom3: UFFAtomParameters) -> float:
    """Return the force constant for an angle bend term."""

    cos_theta0 = math.cos(theta0)
    r12 = calc_bond_rest_length(bond_order12, atom1, atom2)
    r23 = calc_bond_rest_length(bond_order23, atom2, atom3)
    r13_sq = (r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * cos_theta0)
    r13 = math.sqrt(max(r13_sq, 1.0e-12))
    beta = 2.0 * UFF_G / max(r12 * r23, 1.0e-12)
    prefactor = beta * atom1.Z1 * atom3.Z1 / max(r13**5, 1.0e-12)
    r_term = r12 * r23
    inner = 3.0 * r_term * (1.0 - cos_theta0**2) - r13_sq * cos_theta0
    return prefactor * r_term * inner


def calc_angle_c_terms(theta0: float) -> Tuple[float, float, float]:
    """Return the (C0, C1, C2) coefficients used for standard angle terms."""

    sin_theta0 = math.sin(theta0)
    sin_sq = max(sin_theta0 * sin_theta0, 1.0e-8)
    c2 = 1.0 / (4.0 * sin_sq)
    cos_theta0 = math.cos(theta0)
    c1 = -4.0 * c2 * cos_theta0
    c0 = c2 * (2.0 * cos_theta0 * cos_theta0 + 1.0)
    return c0, c1, c2


def torsion_equation17(bond_order23: float, at2: UFFAtomParameters,
                       at3: UFFAtomParameters) -> float:
    return 5.0 * math.sqrt(max(at2.U1, 0.0) * max(at3.U1, 0.0)) * (
        1.0 + 4.18 * math.log(max(bond_order23, 1.0e-3))
    )


def torsion_parameters(bond_order23: float, atomic_num2: int, atomic_num3: int,
                       hyb2: str, hyb3: str, params2: UFFAtomParameters,
                       params3: UFFAtomParameters, end_atom_is_sp2: bool) -> Tuple[float, int, float]:
    """Return ``(force_constant, multiplicity, cos_term)`` for a torsion."""

    hyb2 = hyb2.upper()
    hyb3 = hyb3.upper()
    bond_order23 = max(bond_order23, 1.0e-3)
    if hyb2 == "SP3" and hyb3 == "SP3":
        force_constant = math.sqrt(max(params2.V1, 0.0) * max(params3.V1, 0.0))
        order = 3
        cos_term = -1.0
        if abs(bond_order23 - 1.0) < 1e-6 and atomic_num2 in _GROUP6 and atomic_num3 in _GROUP6:
            v2 = 2.0 if atomic_num2 == 8 else 6.8
            v3 = 2.0 if atomic_num3 == 8 else 6.8
            force_constant = math.sqrt(v2 * v3)
            order = 2
            cos_term = -1.0
    elif hyb2 == "SP2" and hyb3 == "SP2":
        force_constant = torsion_equation17(bond_order23, params2, params3)
        order = 2
        cos_term = 1.0
    else:
        force_constant = 1.0
        order = 6
        cos_term = 1.0
        if abs(bond_order23 - 1.0) < 1e-6:
            is_group6_sp3 = (hyb2 == "SP3" and atomic_num2 in _GROUP6 and atomic_num3 not in _GROUP6) or (
                hyb3 == "SP3" and atomic_num3 in _GROUP6 and atomic_num2 not in _GROUP6
            )
            if is_group6_sp3:
                force_constant = torsion_equation17(bond_order23, params2, params3)
                order = 2
                cos_term = -1.0
            elif end_atom_is_sp2:
                force_constant = 2.0
                order = 3
                cos_term = -1.0
    return force_constant, order, cos_term


def calc_nonbonded_minimum(at1: UFFAtomParameters, at2: UFFAtomParameters) -> float:
    return math.sqrt(at1.x1 * at2.x1)


def calc_nonbonded_depth(at1: UFFAtomParameters, at2: UFFAtomParameters) -> float:
    return math.sqrt(at1.D1 * at2.D1)


def calc_inversion_coefficients_and_force_constant(
    atomic_num: int, is_carbonyl_carbon: bool
) -> Tuple[float, float, float, float]:
    if atomic_num in (6, 7, 8):
        c0 = 1.0
        c1 = -1.0
        c2 = 0.0
        force = 50.0 if (atomic_num == 6 and is_carbonyl_carbon) else 6.0
    else:
        w0 = math.pi / 180.0
        if atomic_num == 15:
            w0 *= 84.4339
        elif atomic_num == 33:
            w0 *= 86.9735
        elif atomic_num == 51:
            w0 *= 87.7047
        elif atomic_num == 83:
            w0 *= 90.0
        else:
            w0 *= 90.0
        c2 = 1.0
        c1 = -4.0 * math.cos(w0)
        c0 = -(c1 * math.cos(w0) + c2 * math.cos(2.0 * w0))
        force = 22.0 / max(c0 + c1 + c2, 1.0e-8)
    force /= 3.0
    return force, c0, c1, c2


def neighbour_matrix(num_atoms: int, bonds: Iterable[Tuple[int, int]]) -> list[list[int]]:
    """Return relationship categories between atom pairs.

    The output matches the encoding used by RDKit: 0 for bonded (1-2),
    1 for 1-3 connections, 2 for 1-4, and 3 otherwise.
    """

    rel = [[3] * num_atoms for _ in range(num_atoms)]
    for i in range(num_atoms):
        rel[i][i] = 0
    adjacency = {i: set() for i in range(num_atoms)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
        rel[i][j] = rel[j][i] = 0
    for i in range(num_atoms):
        for j in adjacency[i]:
            for k in adjacency[j]:
                if k == i:
                    continue
                rel[i][k] = min(rel[i][k], 1)
                rel[k][i] = min(rel[k][i], 1)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if rel[i][j] <= 1:
                continue
            for k in adjacency[i]:
                if rel[k][j] == 1:
                    rel[i][j] = rel[j][i] = min(rel[i][j], 2)
                    break
    return rel
