"""PyTorch module implementing the UFF energy expression."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .builder import UFFInputs
from .utils import calc_nonbonded_depth, calc_nonbonded_minimum, neighbour_matrix


def _prepare_coords(coords: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if coords is reference:
        prepared = reference
    else:
        prepared = coords.to(
            device=reference.device,
            dtype=reference.dtype,
            non_blocking=True,
        )
    if prepared.dim() == 2:
        if prepared.size(-1) != 3:
            raise ValueError("Coordinates must have shape (N, 3).")
        return prepared.contiguous()
    if prepared.dim() == 3:
        if prepared.size(-1) != 3:
            raise ValueError("Coordinates must have shape (batch, N, 3).")
        return prepared.contiguous()
    raise ValueError("Coordinates must have shape (N, 3) or (batch, N, 3).")


def _select_atoms(coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        return torch.index_select(coords, 0, indices)
    return torch.index_select(coords, 1, indices)


def _pair_vectors(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return (
            torch.empty((coords.shape[0], 0, 3), device=coords.device, dtype=coords.dtype)
            if coords.dim() == 3
            else torch.empty((0, 3), device=coords.device, dtype=coords.dtype)
        )
    first = _select_atoms(coords, index[:, 0])
    second = _select_atoms(coords, index[:, 1])
    return first - second


def _gather(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        shape = (coords.shape[0], 0, 3) if coords.dim() == 3 else (0, 3)
        return torch.empty(shape, device=coords.device, dtype=coords.dtype)
    gathered = _select_atoms(coords, index)
    return gathered.contiguous()


class UFFTorch(nn.Module):
    """Torch module wrapping the Universal Force Field energy."""

    def __init__(self, inputs: UFFInputs):
        super().__init__()
        self.n_atoms = inputs.coordinates.shape[0]
        self.reference_coords = nn.Parameter(
            inputs.coordinates.clone().detach(), requires_grad=True
        )
        self.register_buffer("bond_index", inputs.bond_index)
        self.register_buffer("bond_rest_length", inputs.bond_rest_length)
        self.register_buffer("bond_force_constant", inputs.bond_force_constant)
        self.register_buffer("bond_half_force_constant", 0.5 * inputs.bond_force_constant)
        self.register_buffer("angle_index", inputs.angle_index)
        self.register_buffer("angle_force_constant", inputs.angle_force_constant)
        self.register_buffer("angle_c0", inputs.angle_c0)
        self.register_buffer("angle_c1", inputs.angle_c1)
        self.register_buffer("angle_c2", inputs.angle_c2)
        self.register_buffer("angle_order", inputs.angle_order)
        self.register_buffer(
            "angle_order_float",
            inputs.angle_order.to(dtype=inputs.coordinates.dtype)
            if inputs.angle_order.numel()
            else inputs.angle_order.new_empty((0,), dtype=inputs.coordinates.dtype),
        )
        self.register_buffer("torsion_index", inputs.torsion_index)
        self.register_buffer("torsion_force_constant", inputs.torsion_force_constant)
        self.register_buffer("torsion_order", inputs.torsion_order)
        self.register_buffer("torsion_cos_term", inputs.torsion_cos_term)
        self.register_buffer(
            "torsion_half_force_constant",
            0.5 * inputs.torsion_force_constant,
        )
        self.register_buffer("inversion_index", inputs.inversion_index)
        self.register_buffer("inversion_force_constant", inputs.inversion_force_constant)
        self.register_buffer("inversion_c0", inputs.inversion_c0)
        self.register_buffer("inversion_c1", inputs.inversion_c1)
        self.register_buffer("inversion_c2", inputs.inversion_c2)
        self.register_buffer("nonbond_index", inputs.nonbond_index)
        self.register_buffer("vdw_minimum", inputs.vdw_minimum)
        self.register_buffer("vdw_well_depth", inputs.vdw_well_depth)
        self.register_buffer("vdw_threshold", inputs.vdw_threshold)
        self.register_buffer("vdw_threshold_sq", inputs.vdw_threshold.square())

        self._vdw_distance_multiplier = float(inputs.vdw_distance_multiplier)
        fragment_ids = inputs.fragment_ids
        allow_interfragment = inputs.allow_interfragment_interactions
        if inputs.bond_index.numel():
            bonds = [(int(i), int(j)) for i, j in inputs.bond_index.detach().cpu().tolist()]
        else:
            bonds = []
        relation = neighbour_matrix(self.n_atoms, bonds)
        candidate_pairs: List[tuple[int, int]] = []
        candidate_min: List[float] = []
        candidate_depth: List[float] = []
        for idx_i, params_i in enumerate(inputs.atom_params):
            if params_i is None:
                continue
            for idx_j in range(idx_i + 1, self.n_atoms):
                params_j = inputs.atom_params[idx_j]
                if params_j is None:
                    continue
                if relation[idx_i][idx_j] < 2:
                    continue
                if (
                    not allow_interfragment
                    and fragment_ids is not None
                    and fragment_ids[idx_i] != fragment_ids[idx_j]
                ):
                    continue
                candidate_pairs.append((idx_i, idx_j))
                candidate_min.append(calc_nonbonded_minimum(params_i, params_j))
                candidate_depth.append(calc_nonbonded_depth(params_i, params_j))
        device = self.reference_coords.device
        dtype = self.reference_coords.dtype
        if candidate_pairs:
            candidate_index = torch.tensor(candidate_pairs, device=device, dtype=torch.long)
            candidate_minimum = torch.tensor(candidate_min, device=device, dtype=dtype)
            candidate_depth = torch.tensor(candidate_depth, device=device, dtype=dtype)
        else:
            candidate_index = torch.empty((0, 2), device=device, dtype=torch.long)
            candidate_minimum = torch.empty((0,), device=device, dtype=dtype)
            candidate_depth = torch.empty((0,), device=device, dtype=dtype)
        candidate_threshold = candidate_minimum * self._vdw_distance_multiplier
        candidate_threshold_sq = candidate_threshold.square()
        self.register_buffer("nonbond_candidate_index", candidate_index)
        self.register_buffer("nonbond_candidate_minimum", candidate_minimum)
        self.register_buffer("nonbond_candidate_well_depth", candidate_depth)
        self.register_buffer("nonbond_candidate_threshold", candidate_threshold)
        self.register_buffer("nonbond_candidate_threshold_sq", candidate_threshold_sq)

        with torch.no_grad():
            self._refresh_nonbond_pairs(self.reference_coords)

    def forward(self, coords: Optional[torch.Tensor] = None, *, return_components: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        coords_input = coords if coords is not None else self.reference_coords
        coords = _prepare_coords(coords_input, self.reference_coords)
        with torch.no_grad():
            self._refresh_nonbond_pairs(coords)
        energies: Dict[str, torch.Tensor] = {}
        energies["bond"] = self._bond_energy(coords)
        energies["angle"] = self._angle_energy(coords)
        energies["torsion"] = self._torsion_energy(coords)
        energies["inversion"] = self._inversion_energy(coords)
        energies["vdw"] = self._nonbonded_energy(coords)
        total = sum(energies.values())
        if return_components:
            energies["total"] = total
            return energies
        return total

    def _bond_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.bond_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        diff = _pair_vectors(coords, self.bond_index)
        dist = torch.linalg.norm(diff, dim=-1)
        stretch = dist - self.bond_rest_length
        energy = self.bond_half_force_constant * torch.square(stretch)
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    def _angle_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.angle_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        p1 = _gather(coords, self.angle_index[:, 0])
        p2 = _gather(coords, self.angle_index[:, 1])
        p3 = _gather(coords, self.angle_index[:, 2])
        v1 = p1 - p2
        v2 = p3 - p2
        dot = (v1 * v2).sum(dim=-1)
        d1_sq = (v1 * v1).sum(dim=-1)
        d2_sq = (v2 * v2).sum(dim=-1)
        inv_denom = torch.rsqrt((d1_sq * d2_sq).clamp_min(1e-24))
        cos_theta = torch.clamp(dot * inv_denom, -0.999999, 0.999999)
        cos_sq = torch.square(cos_theta)
        sin_sq = torch.clamp(1.0 - cos_sq, min=1e-12)
        cos2 = cos_sq - sin_sq
        energy_term = self.angle_c0 + self.angle_c1 * cos_theta + self.angle_c2 * cos2
        if self.angle_order.numel() and (self.angle_order > 0).any():
            order_long = self.angle_order
            order_float = self.angle_order_float
            if coords.dim() == 3:
                order_long = order_long.unsqueeze(0).expand_as(cos_theta)
                order_float = order_float.unsqueeze(0).expand_as(cos_theta)
            else:
                order_long = order_long.expand_as(cos_theta)
                order_float = order_float.expand_as(cos_theta)
            mask = order_long > 0
            if mask.any():
                terms = torch.zeros_like(cos_theta)
                terms = torch.where(order_long == 1, -cos_theta, terms)
                terms = torch.where(order_long == 2, cos2, terms)
                terms = torch.where(
                    order_long == 3,
                    cos_theta * (cos_sq - 3.0 * sin_sq),
                    terms,
                )
                cos_sq_sq = torch.square(cos_sq)
                sin_sq_sq = torch.square(sin_sq)
                terms = torch.where(
                    order_long == 4,
                    cos_sq_sq - 6.0 * cos_sq * sin_sq + sin_sq_sq,
                    terms,
                )
                denom = torch.clamp(order_float * order_float, min=1.0)
                replacement = (1.0 - terms) / denom
                energy_term = torch.where(mask, replacement, energy_term)
        energy = self.angle_force_constant * energy_term
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    def _torsion_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.torsion_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        p1 = _gather(coords, self.torsion_index[:, 0])
        p2 = _gather(coords, self.torsion_index[:, 1])
        p3 = _gather(coords, self.torsion_index[:, 2])
        p4 = _gather(coords, self.torsion_index[:, 3])
        r1 = p1 - p2
        r2 = p3 - p2
        r3 = p2 - p3
        r4 = p4 - p3
        t1 = torch.cross(r1, r2, dim=-1)
        t2 = torch.cross(r3, r4, dim=-1)
        d1_sq = (t1 * t1).sum(dim=-1)
        d2_sq = (t2 * t2).sum(dim=-1)
        inv_denom = torch.rsqrt((d1_sq * d2_sq).clamp_min(1e-24))
        cos_phi = torch.clamp((t1 * t2).sum(dim=-1) * inv_denom, -0.999999, 0.999999)
        cos_sq = torch.square(cos_phi)
        sin_sq = torch.clamp(1.0 - cos_sq, min=1e-12)
        order = self.torsion_order
        if coords.dim() == 3:
            order = order.unsqueeze(0).expand_as(cos_phi)
        cos_n_phi = torch.zeros_like(cos_phi)
        if (order == 1).any():
            cos_n_phi = torch.where(order == 1, cos_phi, cos_n_phi)
        if (order == 2).any():
            cos_n_phi = torch.where(order == 2, 1.0 - 2.0 * sin_sq, cos_n_phi)
        if (order == 3).any():
            cos_n_phi = torch.where(
                order == 3,
                cos_phi * (cos_sq - 3.0 * sin_sq),
                cos_n_phi,
            )
        if (order == 4).any():
            sin_sq_sq = torch.square(sin_sq)
            cos_sq_sq = torch.square(cos_sq)
            cos_n_phi = torch.where(
                order == 4,
                cos_sq_sq - 6.0 * cos_sq * sin_sq + sin_sq_sq,
                cos_n_phi,
            )
        if (order == 6).any():
            cos_n_phi = torch.where(
                order == 6,
                1.0 + sin_sq * (-32.0 * sin_sq * sin_sq + 48.0 * sin_sq - 18.0),
                cos_n_phi,
            )
        energy = self.torsion_half_force_constant * (1.0 - self.torsion_cos_term * cos_n_phi)
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    def _inversion_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.inversion_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        i = self.inversion_index[:, 0]
        j = self.inversion_index[:, 1]
        k = self.inversion_index[:, 2]
        l = self.inversion_index[:, 3]
        p_i = _gather(coords, i)
        p_j = _gather(coords, j)
        p_k = _gather(coords, k)
        p_l = _gather(coords, l)
        r_ji = p_i - p_j
        r_jk = p_k - p_j
        r_jl = p_l - p_j
        d_ji = torch.linalg.norm(r_ji, dim=-1)
        d_jk = torch.linalg.norm(r_jk, dim=-1)
        d_jl = torch.linalg.norm(r_jl, dim=-1)
        mask = (d_ji > 1e-8) & (d_jk > 1e-8) & (d_jl > 1e-8)
        safe = mask.to(dtype=coords.dtype)
        r_ji = r_ji / torch.clamp(d_ji.unsqueeze(-1), min=1e-8)
        r_jk = r_jk / torch.clamp(d_jk.unsqueeze(-1), min=1e-8)
        r_jl = r_jl / torch.clamp(d_jl.unsqueeze(-1), min=1e-8)
        n = torch.cross(-r_ji, r_jk, dim=-1)
        n_norm = torch.linalg.norm(n, dim=-1)
        n = n / torch.clamp(n_norm.unsqueeze(-1), min=1e-8)
        cos_y = torch.clamp((n * r_jl).sum(dim=-1), -0.999999, 0.999999)
        cos_y_sq = torch.square(cos_y)
        sin_y = torch.sqrt(torch.clamp(1.0 - cos_y_sq, min=1e-12))
        sin_y_sq = torch.square(sin_y)
        cos_2w = 2.0 * sin_y_sq - 1.0
        energy = self.inversion_force_constant * (
            self.inversion_c0 + self.inversion_c1 * sin_y + self.inversion_c2 * cos_2w
        )
        energy = energy * safe
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    def _nonbonded_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.nonbond_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        diff = _pair_vectors(coords, self.nonbond_index)
        dist_sq = (diff * diff).sum(dim=-1)
        inv_dist = torch.rsqrt(dist_sq.clamp_min(1e-12))
        if coords.dim() == 2:
            thresh_sq = self.vdw_threshold_sq
            vdw_min = self.vdw_minimum
            vdw_depth = self.vdw_well_depth
        else:
            thresh_sq = self.vdw_threshold_sq.unsqueeze(0)
            vdw_min = self.vdw_minimum.unsqueeze(0)
            vdw_depth = self.vdw_well_depth.unsqueeze(0)
        valid = (dist_sq > 0) & (dist_sq <= thresh_sq)
        r = vdw_min * inv_dist
        r2 = torch.square(r)
        r4 = torch.square(r2)
        r6 = r4 * r2
        r12 = torch.square(r6)
        energy = vdw_depth * (r12 - 2.0 * r6)
        energy = energy.masked_fill(~valid, 0.0)
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    def _refresh_nonbond_pairs(self, coords: torch.Tensor) -> None:
        if self.nonbond_candidate_index.numel() == 0:
            empty_index = torch.empty((0, 2), device=coords.device, dtype=torch.long)
            empty_float = torch.empty((0,), device=coords.device, dtype=coords.dtype)
            self.nonbond_index = empty_index
            self.vdw_minimum = empty_float
            self.vdw_well_depth = empty_float
            self.vdw_threshold = empty_float
            self.vdw_threshold_sq = empty_float
            return
        candidate_index = self.nonbond_candidate_index
        candidate_min = self.nonbond_candidate_minimum
        candidate_depth = self.nonbond_candidate_well_depth
        candidate_threshold = self.nonbond_candidate_threshold
        candidate_threshold_sq = self.nonbond_candidate_threshold_sq
        diff = _pair_vectors(coords, candidate_index)
        dist_sq = (diff * diff).sum(dim=-1)
        if coords.dim() == 3:
            positive = dist_sq > 0
            threshold_sq = candidate_threshold_sq.unsqueeze(0)
            within = dist_sq <= threshold_sq
            valid = (positive & within).any(dim=0)
        else:
            positive = dist_sq > 0
            within = dist_sq <= candidate_threshold_sq
            valid = positive & within
        if valid.any():
            new_index = candidate_index[valid]
            new_min = candidate_min[valid]
            new_depth = candidate_depth[valid]
            new_threshold = candidate_threshold[valid]
        else:
            new_index = candidate_index.new_empty((0, 2))
            new_min = candidate_min.new_empty((0,))
            new_depth = candidate_depth.new_empty((0,))
            new_threshold = candidate_threshold.new_empty((0,))
        self.nonbond_index = new_index
        self.vdw_minimum = new_min
        self.vdw_well_depth = new_depth
        self.vdw_threshold = new_threshold
        self.vdw_threshold_sq = new_threshold.square()
