"""PyTorch module implementing the UFF energy expression (radius_graph + torch_sparse fast-path)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch_cluster import radius_graph  # REQUIRED
from torch_sparse import SparseTensor

from .builder import UFFInputs


# ------------------------------ helpers ------------------------------

def _prepare_coords(coords: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Ensure coords is on reference's device/dtype and (N,3) or (B,N,3) contiguous."""
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
    """Slice coords by atom indices for either (N,3) or (B,N,3)."""
    if coords.dim() == 2:
        return torch.index_select(coords, 0, indices)
    return torch.index_select(coords, 1, indices)


def _pair_vectors(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Return r_ij = r_i - r_j given pair index (P,2). Handles batched coords."""
    if index.numel() == 0:
        if coords.dim() == 3:
            return torch.empty((coords.shape[0], 0, 3), device=coords.device, dtype=coords.dtype)
        return torch.empty((0, 3), device=coords.device, dtype=coords.dtype)
    first = _select_atoms(coords, index[:, 0])
    second = _select_atoms(coords, index[:, 1])
    return first - second


def _gather(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather atom positions by index for either (N,3) or (B,N,3)."""
    if index.numel() == 0:
        shape = (coords.shape[0], 0, 3) if coords.dim() == 3 else (0, 3)
        return torch.empty(shape, device=coords.device, dtype=coords.dtype)
    gathered = _select_atoms(coords, index)
    return gathered.contiguous()


def _extract_vdw_params(
    atom_params: List[Optional[object]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract per-atom VdW parameters (R*_i, D_i) from UFFAtomParameters list.
    Returns: (valid_mask [N], R [N], D [N])
    Tries common field names: vdw_minimum/r_vdw and vdw_well_depth/epsilon.
    Falls back to the canonical RDKit column names (x1, D1).
    """
    n = len(atom_params)
    valid = torch.zeros(n, device=device, dtype=torch.bool)
    R = torch.zeros(n, device=device, dtype=dtype)
    D = torch.zeros(n, device=device, dtype=dtype)
    for idx, p in enumerate(atom_params):
        if p is None:
            continue
        r_i = getattr(p, "vdw_minimum", None)
        if r_i is None:
            r_i = getattr(p, "r_vdw", None)
        if r_i is None:
            r_i = getattr(p, "x1", None)
        d_i = getattr(p, "vdw_well_depth", None)
        if d_i is None:
            d_i = getattr(p, "epsilon", None)
        if d_i is None:
            d_i = getattr(p, "D1", None)
        if r_i is None or d_i is None:
            continue
        R[idx] = float(r_i)
        D[idx] = float(d_i)
        valid[idx] = True
    return valid, R, D


# ------------------------------ main module ------------------------------


class UFFTorch(nn.Module):
    """Torch module wrapping the Universal Force Field energy (nonbond via radius_graph; fast 1–2/1–3 mask)."""

    def __init__(self, inputs: UFFInputs, *, refresh_chunk_size: int = 8):
        super().__init__()

        # --- basic geometry & reference coords ---
        self.n_atoms = int(inputs.coordinates.shape[0])
        self.reference_coords = nn.Parameter(
            inputs.coordinates.clone().detach(), requires_grad=True
        )

        # --- bonded / angle / torsion / inversion buffers ---
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
        self.register_buffer("torsion_half_force_constant", 0.5 * inputs.torsion_force_constant)

        self.register_buffer("inversion_index", inputs.inversion_index)
        self.register_buffer("inversion_force_constant", inputs.inversion_force_constant)
        self.register_buffer("inversion_c0", inputs.inversion_c0)
        self.register_buffer("inversion_c1", inputs.inversion_c1)
        self.register_buffer("inversion_c2", inputs.inversion_c2)

        # --- vdW settings & per-atom params ---
        self._vdw_distance_multiplier = float(inputs.vdw_distance_multiplier)
        device = self.reference_coords.device
        dtype = self.reference_coords.dtype

        if inputs.fragment_ids is not None:
            self.register_buffer(
                "_fragment_ids",
                torch.tensor(inputs.fragment_ids, device=device, dtype=torch.long),
                persistent=False,
            )
        else:
            self._fragment_ids = None  # type: ignore[assignment]
        self._allow_interfragment = bool(inputs.allow_interfragment_interactions)

        valid_mask, atom_R, atom_D = _extract_vdw_params(inputs.atom_params, device, dtype)
        self.register_buffer("vdw_atom_valid", valid_mask)
        self.register_buffer("vdw_atom_R", atom_R)
        self.register_buffer("vdw_atom_D", atom_D)

        # tuning for batched refresh
        self.refresh_chunk_size = int(refresh_chunk_size)

        # --- build candidate nonbond pairs using optimized path ---
        cand_index, cand_Rij, cand_Dij, cand_cutoff = self._build_nonbond_candidates(
            coords=inputs.coordinates.to(device=device, dtype=dtype),
            bond_index=inputs.bond_index,
        )

        # register candidate buffers
        self.register_buffer("nonbond_candidate_index", cand_index)                   # (P,2)
        self.register_buffer("nonbond_candidate_minimum", cand_Rij)                   # (P,)
        self.register_buffer("nonbond_candidate_well_depth", cand_Dij)                # (P,)
        self.register_buffer("nonbond_candidate_threshold", cand_cutoff)              # (P,)
        self.register_buffer("nonbond_candidate_threshold_sq", cand_cutoff.square())  # (P,)

        # --- current active nonbond buffers (populated by _refresh_nonbond_pairs) ---
        self.register_buffer("nonbond_index", torch.empty((0, 2), device=device, dtype=torch.long))
        self.register_buffer("vdw_minimum", torch.empty((0,), device=device, dtype=dtype))
        self.register_buffer("vdw_well_depth", torch.empty((0,), device=device, dtype=dtype))
        self.register_buffer("vdw_threshold", torch.empty((0,), device=device, dtype=dtype))
        self.register_buffer("vdw_threshold_sq", torch.empty((0,), device=device, dtype=dtype))

        # --- initialize nonbond list from initial coords (reference + optional batch) ---
        if inputs.batch_coordinates is not None:
            batch_coords = inputs.batch_coordinates.to(device=device, dtype=dtype, non_blocking=True)
            base = self.reference_coords.detach().unsqueeze(0)  # (1,N,3)
            initial_coords = torch.cat([base, batch_coords], dim=0)  # (B',N,3)
        else:
            initial_coords = self.reference_coords  # (N,3)
        with torch.no_grad():
            self._refresh_nonbond_pairs(initial_coords)

    # -------------------------- public API --------------------------

    def forward(
        self,
        coords: Optional[torch.Tensor] = None,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
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

    # -------------------------- energy terms --------------------------

    def _bond_energy(self, coords: torch.Tensor) -> torch.Tensor:
        if self.bond_index.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        diff = _pair_vectors(coords, self.bond_index)   # (..., Pbond, 3)
        dist = torch.linalg.norm(diff, dim=-1)          # (..., Pbond)
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

        # special order handling (sp, sp2, sp3, ...)
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
        sin_sq = torch.clamp(1.0 - cos_sq, min=1.0e-12)

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
                order == 3, cos_phi * (cos_sq - 3.0 * sin_sq), cos_n_phi
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

        diff = _pair_vectors(coords, self.nonbond_index)        # (..., Pnb, 3)
        dist_sq = (diff * diff).sum(dim=-1)                     # (..., Pnb)
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

        # Lennard-Jones 12-6 using UFF mixing: E = D_ij [ (Rij/r)^12 - 2 (Rij/r)^6 ]
        r = vdw_min * inv_dist
        r2 = torch.square(r)
        r4 = torch.square(r2)
        r6 = r4 * r2
        r12 = torch.square(r6)
        energy = vdw_depth * (r12 - 2.0 * r6)
        energy = energy.masked_fill(~valid, 0.0)

        return energy.sum() if coords.dim() == 2 else energy.sum(dim=-1)

    # -------------------------- dynamic activation --------------------------

    def _refresh_nonbond_pairs(self, coords: torch.Tensor) -> None:
        """
        Activate a subset of candidate pairs based on current coords:
        keep pairs with any-batch distance <= pair-specific cutoff and > 0.
        Chunked across batch to reduce memory (step 5).
        """
        if self.nonbond_candidate_index.numel() == 0:
            empty_index = torch.empty((0, 2), device=coords.device, dtype=torch.long)
            empty_float = torch.empty((0,), device=coords.device, dtype=coords.dtype)
            self.nonbond_index = empty_index
            self.vdw_minimum = empty_float
            self.vdw_well_depth = empty_float
            self.vdw_threshold = empty_float
            self.vdw_threshold_sq = empty_float
            return

        cand_index = self.nonbond_candidate_index
        cand_min = self.nonbond_candidate_minimum
        cand_depth = self.nonbond_candidate_well_depth
        cand_thresh = self.nonbond_candidate_threshold
        cand_thresh_sq = self.nonbond_candidate_threshold_sq

        # Batched case: process in chunks along batch dimension
        if coords.dim() == 3:
            B = coords.shape[0]
            Pc = cand_index.size(0)
            valid = torch.zeros(Pc, dtype=torch.bool, device=coords.device)

            step = max(1, self.refresh_chunk_size)
            for s in range(0, B, step):
                e = min(B, s + step)
                c = coords[s:e]  # (b,N,3)
                diff = _pair_vectors(c, cand_index)          # (b,Pc,3)
                d2 = (diff * diff).sum(dim=-1)               # (b,Pc)
                hit = (d2 > 0) & (d2 <= cand_thresh_sq.unsqueeze(0))
                valid |= hit.any(dim=0)

        else:
            diff = _pair_vectors(coords, cand_index)          # (Pc,3)
            d2 = (diff * diff).sum(dim=-1)                    # (Pc,)
            positive = d2 > 0
            within = d2 <= cand_thresh_sq
            valid = positive & within

        if valid.any():
            self.nonbond_index = cand_index[valid]
            self.vdw_minimum = cand_min[valid]
            self.vdw_well_depth = cand_depth[valid]
            self.vdw_threshold = cand_thresh[valid]
            self.vdw_threshold_sq = cand_thresh_sq[valid]
        else:
            self.nonbond_index = cand_index.new_empty((0, 2))
            self.vdw_minimum = cand_min.new_empty((0,))
            self.vdw_well_depth = cand_depth.new_empty((0,))
            self.vdw_threshold = cand_thresh.new_empty((0,))
            self.vdw_threshold_sq = cand_thresh_sq.new_empty((0,))

    # -------------------------- GPU 1–2/1–3 blocked keys --------------------------

    def _blocked_keys_12_13_gpu(self, bond_index: torch.Tensor, n: int) -> torch.Tensor:
        """
        Build sorted 1D keys (i*n + j) to EXCLUDE pairs (upper triangle only):
        - 1–2 (direct bonds)
        - 1–3 (neighbors-of-neighbors)
        Implemented with torch_sparse.SparseTensor on device.
        """
        if bond_index.numel() == 0:
            return bond_index.new_empty((0,), dtype=torch.long)

        row = bond_index[:, 0].to(torch.long)
        col = bond_index[:, 1].to(torch.long)

        # Undirected adjacency (no self loop)
        A = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        A = A.set_diag(0).to_symmetric()

        # 2-hop for 1–3 pairs
        A2 = A @ A
        A2 = A2.set_diag(0)

        # Combine 1–2 and 1–3; coalesce to unique indices
        B = (A + A2).coalesce()
        br, bc, _ = B.coo()

        # Upper triangle only to avoid duplicates
        mask = br < bc
        br = br[mask]
        bc = bc[mask]

        keys = br * n + bc
        return torch.sort(keys).values

    # -------------------------- candidate build --------------------------

    def _build_nonbond_candidates(
        self,
        coords: torch.Tensor,         # (N,3)
        bond_index: torch.Tensor,     # (E,2) or empty
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build candidate nonbond pairs with a broad geometric prefilter, then
        apply (i) atom validity, (ii) inter-fragment policy at radius_graph stage,
        (iii) 1–2/1–3 exclusions (GPU), and (iv) pair-specific cutoff prefilter.

        Returns:
            (cand_index [P,2], Rij [P], Dij [P], cutoff [P])
        where Rij = sqrt(R*_i R*_j), Dij = sqrt(D_i D_j),
              cutoff = m * Rij  (m = vdw_distance_multiplier).
        """
        device = coords.device
        dtype = coords.dtype
        n = self.n_atoms

        # global safe radius: r_glob = 2 * m * max(R*)
        if self.vdw_atom_valid.any():
            Rmax = self.vdw_atom_R[self.vdw_atom_valid].max()
        else:
            Rmax = torch.tensor(0.0, device=device, dtype=dtype)
        r_glob = float(2.0 * self._vdw_distance_multiplier * torch.clamp(
            Rmax, min=torch.finfo(dtype).eps
        ))

        # Step 1. geometric neighbor graph with optional fragment batching
        batch = None
        if (not self._allow_interfragment) and (self._fragment_ids is not None):
            batch = self._fragment_ids
        edge_index = radius_graph(coords, r=r_glob, loop=False, max_num_neighbors=1024, batch=batch)
        src, dst = edge_index[0], edge_index[1]

        # Step 2. make undirected, keep i<j (unique-by-construction; no torch.unique)
        i = torch.minimum(src, dst)
        j = torch.maximum(src, dst)
        keep = i < j
        i, j = i[keep], j[keep]

        # Early exit
        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            return empty_idx, empty_f, empty_f, empty_f

        # Step 3. per-atom validity (inter-fragment already handled via batch above)
        pair_ok = self.vdw_atom_valid[i] & self.vdw_atom_valid[j]
        i, j = i[pair_ok], j[pair_ok]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            return empty_idx, empty_f, empty_f, empty_f

        # Step 4. 1–2 / 1–3 exclusions (GPU, sorted-bucketize)
        if bond_index.numel():
            blocked = self._blocked_keys_12_13_gpu(bond_index.to(device), n)
            if blocked.numel():
                keys = i * n + j
                order = torch.argsort(keys)
                keys_sorted = keys[order]
                # torch.bucketize(..., right=True) returns the upper-bound index so
                # that ``pos - 1`` points at the candidate on equality.  Using the
                # default ``right=False`` would place exact matches at index 0 and
                # skip them because ``pos > 0`` would be false.
                pos = torch.bucketize(keys_sorted, blocked, right=True)  # blocked is sorted
                hit = (pos > 0) & (blocked[pos - 1] == keys_sorted)  # membership
                keep_sorted = ~hit

                keep_mask = torch.empty_like(keep_sorted)
                keep_mask[order] = keep_sorted

                i, j = i[keep_mask], j[keep_mask]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            return empty_idx, empty_f, empty_f, empty_f

        # Step 5. pairwise parameters & pair-specific cutoff (+ initial distance prefilter)
        Rij = torch.sqrt(torch.clamp(self.vdw_atom_R[i] * self.vdw_atom_R[j], min=0.0))  # (P,)
        Dij = torch.sqrt(torch.clamp(self.vdw_atom_D[i] * self.vdw_atom_D[j], min=0.0))  # (P,)
        cutoff = self._vdw_distance_multiplier * Rij

        # Prefilter by actual distance <= cutoff to shrink candidate pool further
        d2 = (coords[i] - coords[j]).pow(2).sum(dim=-1)
        keep2 = d2 <= cutoff.pow(2)
        i, j = i[keep2], j[keep2]
        Rij = Rij[keep2]
        Dij = Dij[keep2]
        cutoff = cutoff[keep2]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            return empty_idx, empty_f, empty_f, empty_f

        # Step 6. sort candidates by (i*n + j) to improve gather locality
        keys = i * n + j
        order = torch.argsort(keys)
        keys_sorted = keys[order]
        i_sorted = i[order]
        j_sorted = j[order]
        Rij_sorted = Rij[order]
        Dij_sorted = Dij[order]
        cutoff_sorted = cutoff[order]

        # radius_graph emits both directions (i,j) and (j,i); after enforcing
        # i<j we still retain duplicate entries.  Drop them while preserving the
        # original sorted order so we match RDKit's single-contribution pairs.
        if keys_sorted.numel():
            dedup = torch.ones_like(keys_sorted, dtype=torch.bool)
            dedup[1:] = keys_sorted[1:] != keys_sorted[:-1]
            i = i_sorted[dedup]
            j = j_sorted[dedup]
            Rij = Rij_sorted[dedup]
            Dij = Dij_sorted[dedup]
            cutoff = cutoff_sorted[dedup]
        else:
            i = i_sorted
            j = j_sorted
            Rij = Rij_sorted
            Dij = Dij_sorted
            cutoff = cutoff_sorted

        cand_index = torch.stack([i, j], dim=1).contiguous()
        return cand_index, Rij, Dij, cutoff
        
    def _refresh_nonbond_candidates(self, coords: torch.Tensor) -> None:
        """
        Rebuild nonbond_candidate_* from coords.
        - coords: (N, 3) 또는 (B, N, 3)
        - (B,N,3)인 경우, 배치 전체에서 등장하는 (i,j) 페어의 union을 만들어 후보 풀을 갱신한다.
        - 이후 forward()에서 _refresh_nonbond_pairs가 이 후보 풀을 기반으로
          실제 활성 nonbond_index를 coords에 맞게 다시 뽑는다.
        """
        # coords를 reference 기준으로 정리 (이미 있는 헬퍼 재사용)
        coords = _prepare_coords(coords, self.reference_coords)
        device = coords.device
        dtype = coords.dtype
        n = self.n_atoms

        # 단일 샘플이면 기존 _build_nonbond_candidates 한 번 쓰고 끝
        if coords.dim() == 2:
            cand_index, Rij, Dij, cutoff = self._build_nonbond_candidates(
                coords=coords,
                bond_index=self.bond_index,
            )
            # candidate 버퍼 갱신
            self.nonbond_candidate_index = cand_index
            self.nonbond_candidate_minimum = Rij
            self.nonbond_candidate_well_depth = Dij
            self.nonbond_candidate_threshold = cutoff
            self.nonbond_candidate_threshold_sq = cutoff.square()
            return

        # 배치 처리: coords.shape == (B, N, 3)
        if coords.dim() != 3:
            raise ValueError("coords must be (N,3) or (B,N,3)")

        B, N, _ = coords.shape
        if N != n:
            raise ValueError(f"coords second dim {N} must match n_atoms {n}")

        # ---------- 1. global safe radius ----------
        if self.vdw_atom_valid.any():
            Rmax = self.vdw_atom_R[self.vdw_atom_valid].max()
        else:
            Rmax = torch.tensor(0.0, device=device, dtype=dtype)
        r_glob = float(
            2.0 * self._vdw_distance_multiplier * torch.clamp(
                Rmax, min=torch.finfo(dtype).eps
            )
        )

        # ---------- 2. flatten coords + batch label ----------
        coords_flat = coords.reshape(B * N, 3)  # (B*N, 3)

        # 배치 라벨: (batch_id, fragment_id)를 하나의 정수로 encode해서
        # fragment 간, batch 간 edge가 섞이지 않도록 한다.
        if (not self._allow_interfragment) and (self._fragment_ids is not None):
            frag = self._fragment_ids.to(device=device)          # (N,)
            frag_tiled = frag.unsqueeze(0).expand(B, N).reshape(-1)  # (B*N,)
            F = int(frag_tiled.max().item()) + 1
            batch_graph = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
            batch = batch_graph * F + frag_tiled
        else:
            batch = torch.arange(B, device=device).repeat_interleave(N)

        edge_index = radius_graph(
            coords_flat,
            r=r_glob,
            loop=False,
            max_num_neighbors=1024,
            batch=batch,
        )
        src, dst = edge_index[0], edge_index[1]

        if src.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            self.nonbond_candidate_index = empty_idx
            self.nonbond_candidate_minimum = empty_f
            self.nonbond_candidate_well_depth = empty_f
            self.nonbond_candidate_threshold = empty_f
            self.nonbond_candidate_threshold_sq = empty_f
            return

        # ---------- 3. (i,j)를 base 인덱스로 내려서 i<j만 유지 ----------
        # flatten index → 공통 atom index (0..N-1)
        i_base = src % n
        j_base = dst % n

        i = torch.minimum(i_base, j_base)
        j = torch.maximum(i_base, j_base)
        keep = i < j
        i, j = i[keep], j[keep]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            self.nonbond_candidate_index = empty_idx
            self.nonbond_candidate_minimum = empty_f
            self.nonbond_candidate_well_depth = empty_f
            self.nonbond_candidate_threshold = empty_f
            self.nonbond_candidate_threshold_sq = empty_f
            return

        # ---------- 4. per-atom validity ----------
        pair_ok = self.vdw_atom_valid[i] & self.vdw_atom_valid[j]
        i, j = i[pair_ok], j[pair_ok]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            self.nonbond_candidate_index = empty_idx
            self.nonbond_candidate_minimum = empty_f
            self.nonbond_candidate_well_depth = empty_f
            self.nonbond_candidate_threshold = empty_f
            self.nonbond_candidate_threshold_sq = empty_f
            return

        # ---------- 5. 1–2 / 1–3 제외 (기존 blocked 키 재사용) ----------
        if self.bond_index.numel():
            blocked = self._blocked_keys_12_13_gpu(self.bond_index.to(device), n)
            if blocked.numel():
                keys = i * n + j
                order = torch.argsort(keys)
                keys_sorted = keys[order]

                pos = torch.bucketize(keys_sorted, blocked, right=True)
                hit = (pos > 0) & (blocked[pos - 1] == keys_sorted)
                keep_sorted = ~hit

                keep_mask = torch.empty_like(keep_sorted)
                keep_mask[order] = keep_sorted

                i = i[keep_mask]
                j = j[keep_mask]

        if i.numel() == 0:
            empty_idx = torch.empty((0, 2), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=dtype)
            self.nonbond_candidate_index = empty_idx
            self.nonbond_candidate_minimum = empty_f
            self.nonbond_candidate_well_depth = empty_f
            self.nonbond_candidate_threshold = empty_f
            self.nonbond_candidate_threshold_sq = empty_f
            return

        # ---------- 6. pair별 UFF 파라미터 & cutoff ----------
        Rij = torch.sqrt(torch.clamp(self.vdw_atom_R[i] * self.vdw_atom_R[j], min=0.0))  # (P,)
        Dij = torch.sqrt(torch.clamp(self.vdw_atom_D[i] * self.vdw_atom_D[j], min=0.0))  # (P,)
        cutoff = self._vdw_distance_multiplier * Rij                                      # (P,)

        # 여기서는 per-pair cutoff로 다시 거리 prefilter까지 넣으려면
        # 배치 전체에서 min distance를 구해야 해서 scatter가 좀 필요함.
        # 일단 r_glob로 이미 한번 잘라졌으니, 여기서는 prefilter 생략하고
        # 정렬 + 중복제거만 해도 충분히 candidate 수가 제어될 가능성이 큼.

        # ---------- 7. (i*n + j) 기준 정렬 + 중복 제거 ----------
        keys = i * n + j
        order = torch.argsort(keys)
        keys_sorted = keys[order]
        i_sorted = i[order]
        j_sorted = j[order]
        Rij_sorted = Rij[order]
        Dij_sorted = Dij[order]
        cutoff_sorted = cutoff[order]

        if keys_sorted.numel():
            dedup = torch.ones_like(keys_sorted, dtype=torch.bool)
            dedup[1:] = keys_sorted[1:] != keys_sorted[:-1]

            i_final = i_sorted[dedup]
            j_final = j_sorted[dedup]
            Rij_final = Rij_sorted[dedup]
            Dij_final = Dij_sorted[dedup]
            cutoff_final = cutoff_sorted[dedup]
        else:
            i_final = i_sorted
            j_final = j_sorted
            Rij_final = Rij_sorted
            Dij_final = Dij_sorted
            cutoff_final = cutoff_sorted

        cand_index = torch.stack([i_final, j_final], dim=1).contiguous()

        # ---------- 8. candidate 버퍼 갱신 ----------
        self.nonbond_candidate_index = cand_index
        self.nonbond_candidate_minimum = Rij_final
        self.nonbond_candidate_well_depth = Dij_final
        self.nonbond_candidate_threshold = cutoff_final
        self.nonbond_candidate_threshold_sq = cutoff_final.square()
