"""PyTorch module implementing the UFF energy expression."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .builder import UFFInputs


def _prepare_coords(coords: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        return coords
    if coords.dim() == 3:
        return coords
    raise ValueError("Coordinates must have shape (N, 3) or (batch, N, 3).")


def _pair_vectors(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return torch.empty((coords.shape[0], 0, 3), device=coords.device, dtype=coords.dtype) if coords.dim() == 3 else torch.empty((0, 3), device=coords.device, dtype=coords.dtype)
    if coords.dim() == 2:
        return coords[index[:, 0]] - coords[index[:, 1]]
    else:
        return coords[:, index[:, 0], :] - coords[:, index[:, 1], :]


def _gather(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        shape = (coords.shape[0], 0, 3) if coords.dim() == 3 else (0, 3)
        return torch.empty(shape, device=coords.device, dtype=coords.dtype)
    if coords.dim() == 2:
        return coords[index]
    return coords[:, index, :]


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
        self.register_buffer("angle_index", inputs.angle_index)
        self.register_buffer("angle_force_constant", inputs.angle_force_constant)
        self.register_buffer("angle_c0", inputs.angle_c0)
        self.register_buffer("angle_c1", inputs.angle_c1)
        self.register_buffer("angle_c2", inputs.angle_c2)
        self.register_buffer("angle_order", inputs.angle_order)
        self.register_buffer("torsion_index", inputs.torsion_index)
        self.register_buffer("torsion_force_constant", inputs.torsion_force_constant)
        self.register_buffer("torsion_order", inputs.torsion_order)
        self.register_buffer("torsion_cos_term", inputs.torsion_cos_term)
        self.register_buffer("inversion_index", inputs.inversion_index)
        self.register_buffer("inversion_force_constant", inputs.inversion_force_constant)
        self.register_buffer("inversion_c0", inputs.inversion_c0)
        self.register_buffer("inversion_c1", inputs.inversion_c1)
        self.register_buffer("inversion_c2", inputs.inversion_c2)
        self.register_buffer("nonbond_index", inputs.nonbond_index)
        self.register_buffer("vdw_minimum", inputs.vdw_minimum)
        self.register_buffer("vdw_well_depth", inputs.vdw_well_depth)
        self.register_buffer("vdw_threshold", inputs.vdw_threshold)

    def forward(self, coords: Optional[torch.Tensor] = None, *, return_components: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        coords_input = coords if coords is not None else self.reference_coords
        coords = _prepare_coords(coords_input, self.reference_coords)
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
        dist = torch.sqrt((diff * diff).sum(dim=-1))
        stretch = dist - self.bond_rest_length
        energy = 0.5 * self.bond_force_constant * stretch * stretch
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
        sin_sq = torch.clamp(1.0 - cos_theta * cos_theta, min=1e-12)
        cos2 = cos_theta * cos_theta - sin_sq
        energy_term = self.angle_c0 + self.angle_c1 * cos_theta + self.angle_c2 * cos2
        if (self.angle_order > 0).any():
            order = self.angle_order.to(coords.dtype)
            if coords.dim() == 3:
                order = order.unsqueeze(0)
            order = order.expand_as(cos_theta)
            mask = order > 0
            if mask.any():
                cos_sq = cos_theta * cos_theta
                terms = torch.zeros_like(cos_theta)
                terms = torch.where(order == 1, -cos_theta, terms)
                terms = torch.where(order == 2, cos2, terms)
                terms = torch.where(
                    order == 3,
                    cos_theta * (cos_sq - 3.0 * sin_sq),
                    terms,
                )
                terms = torch.where(
                    order == 4,
                    cos_sq * cos_sq - 6.0 * cos_sq * sin_sq + sin_sq * sin_sq,
                    terms,
                )
                denom = torch.clamp(order * order, min=1.0)
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
        sin_sq = torch.clamp(1.0 - cos_phi * cos_phi, min=1e-12)
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
                cos_phi * (cos_phi * cos_phi - 3.0 * sin_sq),
                cos_n_phi,
            )
        if (order == 4).any():
            cos_sq = cos_phi * cos_phi
            sin_sq_sq = sin_sq * sin_sq
            cos_n_phi = torch.where(
                order == 4,
                cos_sq * cos_sq - 6.0 * cos_sq * sin_sq + sin_sq_sq,
                cos_n_phi,
            )
        if (order == 6).any():
            cos_n_phi = torch.where(
                order == 6,
                1.0 + sin_sq * (-32.0 * sin_sq * sin_sq + 48.0 * sin_sq - 18.0),
                cos_n_phi,
            )
        energy = 0.5 * self.torsion_force_constant * (1.0 - self.torsion_cos_term * cos_n_phi)
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
        d_ji = torch.sqrt((r_ji * r_ji).sum(dim=-1))
        d_jk = torch.sqrt((r_jk * r_jk).sum(dim=-1))
        d_jl = torch.sqrt((r_jl * r_jl).sum(dim=-1))
        mask = (d_ji > 1e-8) & (d_jk > 1e-8) & (d_jl > 1e-8)
        safe = mask.float()
        r_ji = r_ji / torch.clamp(d_ji.unsqueeze(-1), min=1e-8)
        r_jk = r_jk / torch.clamp(d_jk.unsqueeze(-1), min=1e-8)
        r_jl = r_jl / torch.clamp(d_jl.unsqueeze(-1), min=1e-8)
        n = torch.cross(-r_ji, r_jk, dim=-1)
        n_norm = torch.sqrt((n * n).sum(dim=-1))
        n = n / torch.clamp(n_norm.unsqueeze(-1), min=1e-8)
        cos_y = torch.clamp((n * r_jl).sum(dim=-1), -0.999999, 0.999999)
        sin_y = torch.sqrt(torch.clamp(1.0 - cos_y * cos_y, min=1e-12))
        cos_2w = 2.0 * sin_y * sin_y - 1.0
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
            thresh = self.vdw_threshold
            thresh_sq = thresh * thresh
            vdw_min = self.vdw_minimum
            vdw_depth = self.vdw_well_depth
        else:
            thresh = self.vdw_threshold.unsqueeze(0)
            thresh_sq = thresh * thresh
            vdw_min = self.vdw_minimum.unsqueeze(0)
            vdw_depth = self.vdw_well_depth.unsqueeze(0)
        valid = (dist_sq > 0) & (dist_sq <= thresh_sq)
        r = vdw_min * inv_dist
        r6 = r.pow(6)
        r12 = r6 * r6
        energy = vdw_depth * (r12 - 2.0 * r6)
        energy = energy.masked_fill(~valid, 0.0)
        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)
