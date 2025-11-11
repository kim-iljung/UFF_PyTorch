"""Conversion utilities from :mod:`rdkit` molecules to tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .atom_typing import uff_atom_type
from .parameters import UFFAtomParameters, get_atom_parameters
from .utils import (
    calc_angle_c_terms,
    calc_angle_force_constant,
    calc_bond_force_constant,
    calc_bond_rest_length,
    calc_inversion_coefficients_and_force_constant,
    torsion_parameters,
)


@dataclass
class UFFInputs:
    """A container with tensors describing a UFF system."""

    atom_types: List[str]
    atom_params: List[Optional[UFFAtomParameters]]
    coordinates: torch.Tensor

    # Bonds
    bond_index: torch.Tensor
    bond_rest_length: torch.Tensor
    bond_force_constant: torch.Tensor

    # Angles
    angle_index: torch.Tensor
    angle_force_constant: torch.Tensor
    angle_c0: torch.Tensor
    angle_c1: torch.Tensor
    angle_c2: torch.Tensor
    angle_order: torch.Tensor

    # Torsions
    torsion_index: torch.Tensor
    torsion_force_constant: torch.Tensor
    torsion_order: torch.Tensor
    torsion_cos_term: torch.Tensor

    # Inversions
    inversion_index: torch.Tensor
    inversion_force_constant: torch.Tensor
    inversion_c0: torch.Tensor
    inversion_c1: torch.Tensor
    inversion_c2: torch.Tensor

    # Non-bonded (left empty here; model will populate from candidates)
    nonbond_index: torch.Tensor
    vdw_minimum: torch.Tensor
    vdw_well_depth: torch.Tensor
    vdw_threshold: torch.Tensor

    # Batch coordinates (optional)
    batch_coordinates: Optional[torch.Tensor] = None

    # Fragment info and VDW cutoff control for model-side candidate building
    fragment_ids: Optional[List[int]] = None
    allow_interfragment_interactions: bool = True
    vdw_distance_multiplier: float = 4.0


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


def _build_single_inputs(
    mol,
    *,
    conf_id: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    vdw_distance_multiplier: float,
    ignore_interfragment_interactions: bool,
) -> UFFInputs:
    conf_id, conf = _ensure_conformer(mol, conf_id)
    atom_types, atom_params, _ = _compute_atom_types(mol)

    # Coordinates
    coords = torch.tensor(
        [
            (
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            )
            for i in range(mol.GetNumAtoms())
        ],
        dtype=dtype,
        device=device,
    )

    # Bonds
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

    # Angles
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

    # Torsions
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

    # Inversions
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

    # Fragment labels only (for model-side candidate masking). No pair building here.
    fragment_labels: Optional[List[int]] = None
    if ignore_interfragment_interactions:
        from rdkit.Chem import rdmolops
        frags = rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        fragment_labels = [0] * mol.GetNumAtoms()
        for frag_id, atoms in enumerate(frags):
            for idx in atoms:
                fragment_labels[idx] = frag_id

    # Return with EMPTY non-bonded tensors; model will populate them from candidates.
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
        nonbond_index=torch.empty((0, 2), device=device, dtype=torch.long),
        vdw_minimum=torch.empty((0,), device=device, dtype=dtype),
        vdw_well_depth=torch.empty((0,), device=device, dtype=dtype),
        vdw_threshold=torch.empty((0,), device=device, dtype=dtype),
        fragment_ids=fragment_labels,
        allow_interfragment_interactions=not ignore_interfragment_interactions,
        vdw_distance_multiplier=vdw_distance_multiplier,
    )


def _is_molecule(obj) -> bool:
    return hasattr(obj, "GetNumAtoms") and callable(getattr(obj, "GetNumAtoms"))


def _normalise_conf_ids(
    conf_id: Optional[Union[int, Sequence[Optional[int]]]], n: int
) -> List[Optional[int]]:
    if conf_id is None or isinstance(conf_id, int):
        return [conf_id] * n
    if isinstance(conf_id, Sequence) and not isinstance(conf_id, (str, bytes)):
        if len(conf_id) != n:
            raise ValueError(
                "The number of conformer identifiers must match the number of molecules."
            )
        normalised: List[Optional[int]] = []
        for idx, value in enumerate(conf_id):
            if value is None or isinstance(value, int):
                normalised.append(value)
            else:
                raise TypeError(
                    "Conformer identifiers must be integers or ``None`` values."
                )
        return normalised
    raise TypeError(
        "The ``conf_id`` argument must be an integer, ``None`` or a sequence of those values."
    )


def _ensure_same_topology(reference: UFFInputs, candidate: UFFInputs, index: int) -> None:
    if reference.atom_types != candidate.atom_types:
        raise ValueError(
            "All molecules in the batch must have identical atom typing; "
            f"entry {index} does not match the reference."
        )
    ref_params = reference.atom_params
    cand_params = candidate.atom_params
    if len(ref_params) != len(cand_params):
        raise ValueError(
            "All molecules in the batch must contain the same number of atoms."
        )
    for idx, (ref_param, cand_param) in enumerate(zip(ref_params, cand_params)):
        if (ref_param is None) != (cand_param is None):
            raise ValueError(
                "All molecules in the batch must agree on atom typing; "
                f"atom {idx} in entry {index} differs."
            )
    tensor_fields: Iterable[str] = (
        "bond_index",
        "bond_rest_length",
        "bond_force_constant",
        "angle_index",
        "angle_force_constant",
        "angle_c0",
        "angle_c1",
        "angle_c2",
        "angle_order",
        "torsion_index",
        "torsion_force_constant",
        "torsion_order",
        "torsion_cos_term",
        "inversion_index",
        "inversion_force_constant",
        "inversion_c0",
        "inversion_c1",
        "inversion_c2",
        "nonbond_index",
        "vdw_minimum",
        "vdw_well_depth",
        "vdw_threshold",
    )
    for field in tensor_fields:
        ref_tensor = getattr(reference, field)
        cand_tensor = getattr(candidate, field)
        if ref_tensor.shape != cand_tensor.shape or not torch.equal(ref_tensor, cand_tensor):
            raise ValueError(
                "All molecules in the batch must yield identical UFF tensors; "
                f"field ``{field}`` differs for entry {index}."
            )


def build_uff_inputs(
    mol,
    *,
    conf_id: Optional[Union[int, Sequence[Optional[int]]]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
    vdw_distance_multiplier: float = 4.0,
    ignore_interfragment_interactions: bool = False,
) -> UFFInputs:
    """Generate :class:`UFFInputs` tensors from one or multiple RDKit molecules.

    When ``mol`` is a sequence, the molecules must yield identical UFF topology
    tensors.  The returned instance keeps the tensors from the first entry for
    compatibility and exposes stacked coordinates through the
    :attr:`UFFInputs.batch_coordinates` attribute so they can be passed directly
    to :meth:`UFFTorch.forward`.
    """

    if device is None:
        device = torch.device("cpu")

    if _is_molecule(mol):
        (conf_choice,) = _normalise_conf_ids(conf_id, 1)
        return _build_single_inputs(
            mol,
            conf_id=conf_choice,
            device=device,
            dtype=dtype,
            vdw_distance_multiplier=vdw_distance_multiplier,
            ignore_interfragment_interactions=ignore_interfragment_interactions,
        )

    if isinstance(mol, Sequence) and not isinstance(mol, (str, bytes)):
        mols = list(mol)
        if not mols:
            raise ValueError("The sequence of molecules cannot be empty.")
        if not all(_is_molecule(entry) for entry in mols):
            raise TypeError("All entries must be RDKit molecule objects.")
        conf_ids = _normalise_conf_ids(conf_id, len(mols))
        reference = _build_single_inputs(
            mols[0],
            conf_id=conf_ids[0],
            device=device,
            dtype=dtype,
            vdw_distance_multiplier=vdw_distance_multiplier,
            ignore_interfragment_interactions=ignore_interfragment_interactions,
        )
        batch_coords = [reference.coordinates]
        for idx, (entry, conf_choice) in enumerate(zip(mols[1:], conf_ids[1:]), start=1):
            candidate = _build_single_inputs(
                entry,
                conf_id=conf_choice,
                device=device,
                dtype=dtype,
                vdw_distance_multiplier=vdw_distance_multiplier,
                ignore_interfragment_interactions=ignore_interfragment_interactions,
            )
            _ensure_same_topology(reference, candidate, idx)
            batch_coords.append(candidate.coordinates)
        stacked = torch.stack(batch_coords, dim=0)
        return UFFInputs(
            atom_types=reference.atom_types,
            atom_params=reference.atom_params,
            coordinates=reference.coordinates,
            batch_coordinates=stacked,
            bond_index=reference.bond_index,
            bond_rest_length=reference.bond_rest_length,
            bond_force_constant=reference.bond_force_constant,
            angle_index=reference.angle_index,
            angle_force_constant=reference.angle_force_constant,
            angle_c0=reference.angle_c0,
            angle_c1=reference.angle_c1,
            angle_c2=reference.angle_c2,
            angle_order=reference.angle_order,
            torsion_index=reference.torsion_index,
            torsion_force_constant=reference.torsion_force_constant,
            torsion_order=reference.torsion_order,
            torsion_cos_term=reference.torsion_cos_term,
            inversion_index=reference.inversion_index,
            inversion_force_constant=reference.inversion_force_constant,
            inversion_c0=reference.inversion_c0,
            inversion_c1=reference.inversion_c1,
            inversion_c2=reference.inversion_c2,
            # keep empty non-bonded
            nonbond_index=reference.nonbond_index.new_empty((0, 2)),
            vdw_minimum=reference.vdw_minimum.new_empty((0,)),
            vdw_well_depth=reference.vdw_well_depth.new_empty((0,)),
            vdw_threshold=reference.vdw_threshold.new_empty((0,)),
            fragment_ids=reference.fragment_ids,
            allow_interfragment_interactions=reference.allow_interfragment_interactions,
            vdw_distance_multiplier=reference.vdw_distance_multiplier,
        )

    raise TypeError("The ``mol`` argument must be an RDKit molecule or a sequence of them.")


def merge_uff_inputs(
    left: UFFInputs,
    right: UFFInputs,
    *,
    ignore_interfragment_interactions: bool = False,
    vdw_distance_multiplier: Optional[float] = None,
) -> UFFInputs:
    """Merge two :class:`UFFInputs` objects into a single system."""

    device = left.coordinates.device
    dtype = left.coordinates.dtype

    right_coords = right.coordinates.to(device=device, dtype=dtype, non_blocking=True)
    left_coords = left.coordinates.to(device=device, dtype=dtype, non_blocking=True)
    coordinates = torch.cat([left_coords, right_coords], dim=0)

    left_batch = left.batch_coordinates
    right_batch = right.batch_coordinates
    batch_size: Optional[int] = None
    if left_batch is not None:
        left_batch = left_batch.to(device=device, dtype=dtype, non_blocking=True)
        batch_size = left_batch.shape[0]
    if right_batch is not None:
        right_batch = right_batch.to(device=device, dtype=dtype, non_blocking=True)
        if batch_size is None:
            batch_size = right_batch.shape[0]
        elif batch_size != right_batch.shape[0]:
            raise ValueError("UFFInputs instances use different batch sizes.")
    if batch_size is not None:
        if left_batch is None:
            left_batch = left_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        if right_batch is None:
            right_batch = right_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_coordinates = torch.cat([left_batch, right_batch], dim=1)
    else:
        batch_coordinates = None

    offset = left_coords.shape[0]

    def _match_device(tensor: torch.Tensor, *, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if target_dtype is None:
            return tensor.to(device=device, non_blocking=True)
        return tensor.to(device=device, dtype=target_dtype, non_blocking=True)

    bond_index = torch.cat(
        [
            _match_device(left.bond_index),
            _match_device(right.bond_index) + offset if right.bond_index.numel() else torch.empty((0, 2), device=device, dtype=torch.long),
        ],
        dim=0,
    )
    bond_rest_length = torch.cat(
        [
            _match_device(left.bond_rest_length, target_dtype=dtype),
            _match_device(right.bond_rest_length, target_dtype=dtype),
        ],
        dim=0,
    )
    bond_force_constant = torch.cat(
        [
            _match_device(left.bond_force_constant, target_dtype=dtype),
            _match_device(right.bond_force_constant, target_dtype=dtype),
        ],
        dim=0,
    )

    def _shift_and_cat(
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if left_tensor.numel():
            parts.append(_match_device(left_tensor))
        if right_tensor.numel():
            parts.append(_match_device(right_tensor) + offset)
        if not parts:
            return torch.empty((0, width), device=device, dtype=torch.long)
        return torch.cat(parts, dim=0)

    angle_index = _shift_and_cat(left.angle_index, right.angle_index, 3)
    torsion_index = _shift_and_cat(left.torsion_index, right.torsion_index, 4)
    inversion_index = _shift_and_cat(left.inversion_index, right.inversion_index, 4)

    angle_force_constant = torch.cat(
        [
            _match_device(left.angle_force_constant, target_dtype=dtype),
            _match_device(right.angle_force_constant, target_dtype=dtype),
        ],
        dim=0,
    )
    angle_c0 = torch.cat(
        [
            _match_device(left.angle_c0, target_dtype=dtype),
            _match_device(right.angle_c0, target_dtype=dtype),
        ],
        dim=0,
    )
    angle_c1 = torch.cat(
        [
            _match_device(left.angle_c1, target_dtype=dtype),
            _match_device(right.angle_c1, target_dtype=dtype),
        ],
        dim=0,
    )
    angle_c2 = torch.cat(
        [
            _match_device(left.angle_c2, target_dtype=dtype),
            _match_device(right.angle_c2, target_dtype=dtype),
        ],
        dim=0,
    )
    angle_order = torch.cat(
        [
            _match_device(left.angle_order),
            _match_device(right.angle_order),
        ],
        dim=0,
    )
    torsion_force_constant = torch.cat(
        [
            _match_device(left.torsion_force_constant, target_dtype=dtype),
            _match_device(right.torsion_force_constant, target_dtype=dtype),
        ],
        dim=0,
    )
    torsion_order = torch.cat(
        [
            _match_device(left.torsion_order),
            _match_device(right.torsion_order),
        ],
        dim=0,
    )
    torsion_cos_term = torch.cat(
        [
            _match_device(left.torsion_cos_term, target_dtype=dtype),
            _match_device(right.torsion_cos_term, target_dtype=dtype),
        ],
        dim=0,
    )
    inversion_force_constant = torch.cat(
        [
            _match_device(left.inversion_force_constant, target_dtype=dtype),
            _match_device(right.inversion_force_constant, target_dtype=dtype),
        ],
        dim=0,
    )
    inversion_c0 = torch.cat(
        [
            _match_device(left.inversion_c0, target_dtype=dtype),
            _match_device(right.inversion_c0, target_dtype=dtype),
        ],
        dim=0,
    )
    inversion_c1 = torch.cat(
        [
            _match_device(left.inversion_c1, target_dtype=dtype),
            _match_device(right.inversion_c1, target_dtype=dtype),
        ],
        dim=0,
    )
    inversion_c2 = torch.cat(
        [
            _match_device(left.inversion_c2, target_dtype=dtype),
            _match_device(right.inversion_c2, target_dtype=dtype),
        ],
        dim=0,
    )

    atom_types = list(left.atom_types) + list(right.atom_types)
    atom_params = list(left.atom_params) + list(right.atom_params)

    left_frag = list(left.fragment_ids) if left.fragment_ids is not None else [0] * left_coords.shape[0]
    right_frag_src = list(right.fragment_ids) if right.fragment_ids is not None else [0] * right_coords.shape[0]
    frag_offset = max(left_frag, default=-1) + 1 if left_frag else 0
    right_frag = [frag + frag_offset for frag in right_frag_src]
    fragment_ids = left_frag + right_frag

    # Unify multiplier (keep same logic as before)
    inferred_multipliers: List[float] = []
    for entry in (left, right):
        if hasattr(entry, "vdw_distance_multiplier") and entry.vdw_distance_multiplier is not None:
            inferred_multipliers.append(float(entry.vdw_distance_multiplier))
    if vdw_distance_multiplier is not None:
        multiplier = float(vdw_distance_multiplier)
    elif inferred_multipliers:
        base = inferred_multipliers[0]
        for value in inferred_multipliers[1:]:
            if abs(value - base) > 1e-6:
                raise ValueError("Input UFFInputs instances use incompatible van der Waals cutoffs.")
        multiplier = base
    else:
        multiplier = 4.0

    allow_interfragment = (
        left.allow_interfragment_interactions
        and right.allow_interfragment_interactions
        and not ignore_interfragment_interactions
    )

    # Non-bonded: keep EMPTY; model will derive from candidates on-the-fly.
    nonbond_index = torch.empty((0, 2), device=device, dtype=torch.long)
    vdw_minimum = torch.empty((0,), device=device, dtype=dtype)
    vdw_well_depth = torch.empty((0,), device=device, dtype=dtype)
    vdw_threshold = torch.empty((0,), device=device, dtype=dtype)

    return UFFInputs(
        atom_types=atom_types,
        atom_params=atom_params,
        coordinates=coordinates,
        batch_coordinates=batch_coordinates,
        bond_index=bond_index,
        bond_rest_length=bond_rest_length,
        bond_force_constant=bond_force_constant,
        angle_index=angle_index,
        angle_force_constant=angle_force_constant,
        angle_c0=angle_c0,
        angle_c1=angle_c1,
        angle_c2=angle_c2,
        angle_order=angle_order,
        torsion_index=torsion_index,
        torsion_force_constant=torsion_force_constant,
        torsion_order=torsion_order,
        torsion_cos_term=torsion_cos_term,
        inversion_index=inversion_index,
        inversion_force_constant=inversion_force_constant,
        inversion_c0=inversion_c0,
        inversion_c1=inversion_c1,
        inversion_c2=inversion_c2,
        nonbond_index=nonbond_index,
        vdw_minimum=vdw_minimum,
        vdw_well_depth=vdw_well_depth,
        vdw_threshold=vdw_threshold,
        fragment_ids=fragment_ids,
        allow_interfragment_interactions=allow_interfragment,
        vdw_distance_multiplier=multiplier,
    )
