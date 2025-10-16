"""Conversion utilities from :mod:`rdkit` molecules to tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .atom_typing import uff_atom_type
from .parameters import UFFAtomParameters, get_atom_parameters
from .utils import (
    calc_angle_c_terms,
    calc_angle_force_constant,
    calc_bond_force_constant,
    calc_bond_rest_length,
    calc_inversion_coefficients_and_force_constant,
    calc_nonbonded_depth,
    calc_nonbonded_minimum,
    neighbour_matrix,
    torsion_parameters,
)


@dataclass
class UFFInputs:
    """A container with tensors describing a UFF system."""

    atom_types: List[str]
    atom_params: List[Optional[UFFAtomParameters]]
    coordinates: torch.Tensor
    bond_index: torch.Tensor
    bond_rest_length: torch.Tensor
    bond_force_constant: torch.Tensor
    angle_index: torch.Tensor
    angle_force_constant: torch.Tensor
    angle_c0: torch.Tensor
    angle_c1: torch.Tensor
    angle_c2: torch.Tensor
    angle_order: torch.Tensor
    torsion_index: torch.Tensor
    torsion_force_constant: torch.Tensor
    torsion_order: torch.Tensor
    torsion_cos_term: torch.Tensor
    inversion_index: torch.Tensor
    inversion_force_constant: torch.Tensor
    inversion_c0: torch.Tensor
    inversion_c1: torch.Tensor
    inversion_c2: torch.Tensor
    nonbond_index: torch.Tensor
    vdw_minimum: torch.Tensor
    vdw_well_depth: torch.Tensor
    vdw_threshold: torch.Tensor


def _as_tensor(data: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(data) == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    return torch.tensor(data, device=device, dtype=dtype)


def _as_index_tensor(data: Sequence[Sequence[int]], device: torch.device, width: int) -> torch.Tensor:
    if len(data) == 0:
        return torch.empty((0, width), device=device, dtype=torch.long)
    return torch.tensor(data, device=device, dtype=torch.long)


def _hybridization_to_string(hyb) -> str:
    name = getattr(hyb, "name", None)
    if name is None:
        return str(hyb).upper()
    return name.upper()


def _ensure_conformer(mol, conf_id: Optional[int]):
    if conf_id is None:
        conf_id = -1
    if not mol.GetNumConformers():  # pragma: no cover - depends on RDKit inputs
        raise ValueError("The molecule does not contain any conformers with 3D coordinates.")
    if conf_id < 0:
        conf_id = mol.GetConformer().GetId()
    conf = mol.GetConformer(conf_id)
    return conf_id, conf


def _compute_atom_types(mol) -> Tuple[List[str], List[Optional[UFFAtomParameters]], bool]:
    atom_types: List[str] = []
    params: List[Optional[UFFAtomParameters]] = []
    all_found = True
    for atom in mol.GetAtoms():
        atom_type = uff_atom_type(atom)
        if not atom_type:
            all_found = False
            atom_types.append("")
            params.append(None)
            continue
        atom_types.append(atom_type)
        params.append(get_atom_parameters(atom_type))
    return atom_types, params, all_found


def build_uff_inputs(
    mol,
    *,
    conf_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
    vdw_distance_multiplier: float = 4.0,
    ignore_interfragment_interactions: bool = False,
) -> UFFInputs:
    """Generate :class:`UFFInputs` tensors from an RDKit molecule."""

    if device is None:
        device = torch.device("cpu")
    conf_id, conf = _ensure_conformer(mol, conf_id)
    atom_types, atom_params, _ = _compute_atom_types(mol)

    coords = torch.tensor(
        [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
         for i in range(mol.GetNumAtoms())],
        dtype=dtype,
        device=device,
    )

    bond_index: List[Tuple[int, int]] = []
    bond_rest: List[float] = []
    bond_force: List[float] = []
    bond_orders: Dict[Tuple[int, int], float] = {}

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        params_i = atom_params[i]
        params_j = atom_params[j]
        if params_i is None or params_j is None:
            continue
        order = float(bond.GetBondTypeAsDouble())
        bond_orders[(i, j)] = bond_orders[(j, i)] = order
        r0 = calc_bond_rest_length(order, params_i, params_j)
        k = calc_bond_force_constant(r0, params_i, params_j)
        bond_index.append((i, j))
        bond_rest.append(r0)
        bond_force.append(k)

    angle_index: List[Tuple[int, int, int]] = []
    angle_force: List[float] = []
    angle_c0: List[float] = []
    angle_c1: List[float] = []
    angle_c2: List[float] = []
    angle_order: List[int] = []

    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        params_j = atom_params[j]
        if params_j is None or atom.GetDegree() < 2:
            continue
        neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors() if atom_params[nbr.GetIdx()] is not None]
        neighbours.sort()
        hyb = _hybridization_to_string(atom.GetHybridization())
        if hyb == "SP":
            order_val = 1
        elif hyb == "SP2":
            order_val = 3
        elif hyb == "SP3D2":
            order_val = 4
        else:
            order_val = 0
        for idx_a in range(len(neighbours)):
            i = neighbours[idx_a]
            for idx_b in range(idx_a + 1, len(neighbours)):
                k = neighbours[idx_b]
                params_i = atom_params[i]
                params_k = atom_params[k]
                if params_i is None or params_k is None:
                    continue
                bond_order_ij = bond_orders.get((i, j), 1.0)
                bond_order_jk = bond_orders.get((j, k), 1.0)
                k_force = calc_angle_force_constant(
                    params_j.theta0, bond_order_ij, bond_order_jk, params_i, params_j, params_k
                )
                c0, c1, c2 = calc_angle_c_terms(params_j.theta0)
                angle_index.append((i, j, k))
                angle_force.append(k_force)
                angle_c0.append(c0)
                angle_c1.append(c1)
                angle_c2.append(c2)
                angle_order.append(order_val)

    torsion_index: List[Tuple[int, int, int, int]] = []
    torsion_force: List[float] = []
    torsion_order: List[int] = []
    torsion_cos: List[float] = []

    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        params2 = atom_params[idx1]
        params3 = atom_params[idx2]
        if params2 is None or params3 is None:
            continue
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)
        hyb1 = _hybridization_to_string(atom1.GetHybridization())
        hyb2 = _hybridization_to_string(atom2.GetHybridization())
        if not ({hyb1, hyb2} & {"SP2", "SP3"}):
            continue
        neighbours1 = [nbr.GetIdx() for nbr in atom1.GetNeighbors() if nbr.GetIdx() != idx2]
        neighbours2 = [nbr.GetIdx() for nbr in atom2.GetNeighbors() if nbr.GetIdx() != idx1]
        contributions: List[Tuple[int, int, int, int, float, int, float]] = []
        for b_idx in neighbours1:
            if atom_params[b_idx] is None or b_idx == idx2:
                continue
            for e_idx in neighbours2:
                if atom_params[e_idx] is None or e_idx == idx1 or e_idx == b_idx:
                    continue
                bond_order23 = bond_orders.get((idx1, idx2), 1.0)
                end_atom_is_sp2 = (
                    _hybridization_to_string(mol.GetAtomWithIdx(b_idx).GetHybridization()) == "SP2"
                    or _hybridization_to_string(mol.GetAtomWithIdx(e_idx).GetHybridization()) == "SP2"
                )
                fc, order_val, cos_term = torsion_parameters(
                    bond_order23,
                    atom1.GetAtomicNum(),
                    atom2.GetAtomicNum(),
                    hyb1,
                    hyb2,
                    params2,
                    params3,
                    end_atom_is_sp2,
                )
                contributions.append((b_idx, idx1, idx2, e_idx, fc, order_val, cos_term))
        n = len(contributions)
        if n == 0:
            continue
        for entry in contributions:
            b_idx, c1, c2, e_idx, fc, order_val, cos_term = entry
            torsion_index.append((b_idx, c1, c2, e_idx))
            torsion_force.append(fc / float(n))
            torsion_order.append(order_val)
            torsion_cos.append(cos_term)

    inversion_index: List[Tuple[int, int, int, int]] = []
    inversion_force: List[float] = []
    inversion_c0: List[float] = []
    inversion_c1: List[float] = []
    inversion_c2: List[float] = []
    for atom in mol.GetAtoms():
        idx_center = atom.GetIdx()
        params_center = atom_params[idx_center]
        if params_center is None or atom.GetDegree() != 3:
            continue
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in (6, 7, 8, 15, 33, 51, 83):
            continue
        if atomic_num in (6, 7, 8) and _hybridization_to_string(atom.GetHybridization()) != "SP2":
            continue
        neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors() if atom_params[nbr.GetIdx()] is not None]
        if len(neighbours) != 3:
            continue
        is_carbonyl_carbon = False
        if atomic_num == 6:
            for n_idx in neighbours:
                n_atom = mol.GetAtomWithIdx(n_idx)
                if n_atom.GetAtomicNum() == 8 and _hybridization_to_string(n_atom.GetHybridization()) == "SP2":
                    is_carbonyl_carbon = True
                    break
        force, c0, c1, c2 = calc_inversion_coefficients_and_force_constant(atomic_num, is_carbonyl_carbon)
        orderings = [
            (neighbours[0], idx_center, neighbours[1], neighbours[2]),
            (neighbours[0], idx_center, neighbours[2], neighbours[1]),
            (neighbours[1], idx_center, neighbours[2], neighbours[0]),
        ]
        for ordering in orderings:
            inversion_index.append(ordering)
            inversion_force.append(force)
            inversion_c0.append(c0)
            inversion_c1.append(c1)
            inversion_c2.append(c2)

    bonds = [(i, j) for i, j in bond_index]
    relation = neighbour_matrix(mol.GetNumAtoms(), bonds)
    nonbond_pairs: List[Tuple[int, int]] = []
    vdw_min: List[float] = []
    vdw_depth: List[float] = []
    vdw_thresh: List[float] = []
    fragment_labels: Optional[List[int]] = None
    if ignore_interfragment_interactions:
        from rdkit.Chem import rdmolops

        frags = rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        fragment_labels = [0] * mol.GetNumAtoms()
        for frag_id, atoms in enumerate(frags):
            for idx in atoms:
                fragment_labels[idx] = frag_id
    for i in range(mol.GetNumAtoms()):
        params_i = atom_params[i]
        if params_i is None:
            continue
        for j in range(i + 1, mol.GetNumAtoms()):
            params_j = atom_params[j]
            if params_j is None:
                continue
            if fragment_labels is not None and fragment_labels[i] != fragment_labels[j]:
                continue
            if relation[i][j] < 2:
                continue
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = (pos_i - pos_j).Length()
            minimum = calc_nonbonded_minimum(params_i, params_j)
            if dist > vdw_distance_multiplier * minimum:
                continue
            nonbond_pairs.append((i, j))
            vdw_min.append(minimum)
            vdw_depth.append(calc_nonbonded_depth(params_i, params_j))
            vdw_thresh.append(vdw_distance_multiplier * minimum)

    return UFFInputs(
        atom_types=atom_types,
        atom_params=atom_params,
        coordinates=coords,
        bond_index=_as_index_tensor(bond_index, device, 2),
        bond_rest_length=_as_tensor(bond_rest, device, dtype),
        bond_force_constant=_as_tensor(bond_force, device, dtype),
        angle_index=_as_index_tensor(angle_index, device, 3),
        angle_force_constant=_as_tensor(angle_force, device, dtype),
        angle_c0=_as_tensor(angle_c0, device, dtype),
        angle_c1=_as_tensor(angle_c1, device, dtype),
        angle_c2=_as_tensor(angle_c2, device, dtype),
        angle_order=torch.tensor(angle_order, device=device, dtype=torch.long)
        if angle_order
        else torch.empty((0,), device=device, dtype=torch.long),
        torsion_index=_as_index_tensor(torsion_index, device, 4),
        torsion_force_constant=_as_tensor(torsion_force, device, dtype),
        torsion_order=torch.tensor(torsion_order, device=device, dtype=torch.long)
        if torsion_order
        else torch.empty((0,), device=device, dtype=torch.long),
        torsion_cos_term=_as_tensor(torsion_cos, device, dtype),
        inversion_index=_as_index_tensor(inversion_index, device, 4),
        inversion_force_constant=_as_tensor(inversion_force, device, dtype),
        inversion_c0=_as_tensor(inversion_c0, device, dtype),
        inversion_c1=_as_tensor(inversion_c1, device, dtype),
        inversion_c2=_as_tensor(inversion_c2, device, dtype),
        nonbond_index=_as_index_tensor(nonbond_pairs, device, 2),
        vdw_minimum=_as_tensor(vdw_min, device, dtype),
        vdw_well_depth=_as_tensor(vdw_depth, device, dtype),
        vdw_threshold=_as_tensor(vdw_thresh, device, dtype),
    )
