"""Triton-powered kernels for performance critical operations."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

try:  # pragma: no cover - imported lazily depending on Triton availability
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - Triton is optional
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


def supports_triton_nonbonded(
    coords: Tensor,
    vdw_minimum: Tensor,
    vdw_well_depth: Tensor,
    vdw_threshold_sq: Tensor,
) -> bool:
    """Return ``True`` if the Triton backend can evaluate the non-bonded term."""

    if not _HAS_TRITON:
        return False
    if coords.device.type != "cuda":
        return False
    if coords.dim() not in (2, 3):
        return False
    dtype = coords.dtype
    if dtype not in (torch.float32, torch.float64):
        return False
    if (
        vdw_minimum.dtype != dtype
        or vdw_well_depth.dtype != dtype
        or vdw_threshold_sq.dtype != dtype
    ):
        return False
    return True


if _HAS_TRITON:
    _BLOCK_SIZE = 256

    @triton.jit
    def _nonbonded_forward_kernel(
        coords_ptr,
        index_ptr,
        vdw_min_ptr,
        vdw_depth_ptr,
        thresh_sq_ptr,
        output_ptr,
        n_pairs,
        stride_coords_atom,
        stride_coords_dim,
        stride_index_pair,
        stride_index_component,
        *,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pairs

        index_offsets = offsets * stride_index_pair
        i = tl.load(
            index_ptr + index_offsets + 0 * stride_index_component,
            mask=mask,
            other=0,
        )
        j = tl.load(
            index_ptr + index_offsets + 1 * stride_index_component,
            mask=mask,
            other=0,
        )

        ci0 = tl.load(
            coords_ptr + i * stride_coords_atom + 0 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        ci1 = tl.load(
            coords_ptr + i * stride_coords_atom + 1 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        ci2 = tl.load(
            coords_ptr + i * stride_coords_atom + 2 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj0 = tl.load(
            coords_ptr + j * stride_coords_atom + 0 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj1 = tl.load(
            coords_ptr + j * stride_coords_atom + 1 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj2 = tl.load(
            coords_ptr + j * stride_coords_atom + 2 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )

        diff0 = ci0 - cj0
        diff1 = ci1 - cj1
        diff2 = ci2 - cj2
        dist_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2

        thresh = tl.load(thresh_sq_ptr + offsets, mask=mask, other=0.0)
        valid = mask & (dist_sq > 0) & (dist_sq <= thresh)

        dist_sq = tl.where(valid, dist_sq, 1.0)
        dist_sq = tl.maximum(dist_sq, 1e-12)
        inv_dist = tl.rsqrt(dist_sq)

        vdw_min = tl.load(vdw_min_ptr + offsets, mask=mask, other=0.0)
        depth = tl.load(vdw_depth_ptr + offsets, mask=mask, other=0.0)

        r = vdw_min * inv_dist
        r2 = r * r
        r4 = r2 * r2
        r6 = r4 * r2
        r12 = r6 * r6
        energy = depth * (r12 - 2.0 * r6)
        energy = tl.where(valid, energy, 0.0)
        tl.store(output_ptr + offsets, energy, mask=mask)

    @triton.jit
    def _nonbonded_backward_kernel(
        coords_ptr,
        index_ptr,
        vdw_min_ptr,
        vdw_depth_ptr,
        thresh_sq_ptr,
        grad_coords_ptr,
        n_pairs,
        stride_coords_atom,
        stride_coords_dim,
        stride_index_pair,
        stride_index_component,
        *,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pairs

        index_offsets = offsets * stride_index_pair
        i = tl.load(
            index_ptr + index_offsets + 0 * stride_index_component,
            mask=mask,
            other=0,
        )
        j = tl.load(
            index_ptr + index_offsets + 1 * stride_index_component,
            mask=mask,
            other=0,
        )

        ci0 = tl.load(
            coords_ptr + i * stride_coords_atom + 0 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        ci1 = tl.load(
            coords_ptr + i * stride_coords_atom + 1 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        ci2 = tl.load(
            coords_ptr + i * stride_coords_atom + 2 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj0 = tl.load(
            coords_ptr + j * stride_coords_atom + 0 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj1 = tl.load(
            coords_ptr + j * stride_coords_atom + 1 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )
        cj2 = tl.load(
            coords_ptr + j * stride_coords_atom + 2 * stride_coords_dim,
            mask=mask,
            other=0.0,
        )

        diff0 = ci0 - cj0
        diff1 = ci1 - cj1
        diff2 = ci2 - cj2
        dist_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2

        thresh = tl.load(thresh_sq_ptr + offsets, mask=mask, other=0.0)
        valid = mask & (dist_sq > 0) & (dist_sq <= thresh)

        dist_sq = tl.where(valid, dist_sq, 1.0)
        dist_sq = tl.maximum(dist_sq, 1e-12)
        inv_dist = tl.rsqrt(dist_sq)

        vdw_min = tl.load(vdw_min_ptr + offsets, mask=mask, other=0.0)
        depth = tl.load(vdw_depth_ptr + offsets, mask=mask, other=0.0)

        inv_dist2 = inv_dist * inv_dist
        inv_dist4 = inv_dist2 * inv_dist2
        inv_dist6 = inv_dist4 * inv_dist2
        inv_dist8 = inv_dist4 * inv_dist4

        vdw_min2 = vdw_min * vdw_min
        vdw_min4 = vdw_min2 * vdw_min2
        vdw_min6 = vdw_min4 * vdw_min2
        vdw_min12 = vdw_min6 * vdw_min6

        term = vdw_min12 * inv_dist6 - vdw_min6
        grad_factor = -12.0 * depth * inv_dist8 * term
        grad_factor = tl.where(valid, grad_factor, 0.0)

        grad0 = grad_factor * diff0
        grad1 = grad_factor * diff1
        grad2 = grad_factor * diff2

        tl.atomic_add(
            grad_coords_ptr + i * stride_coords_atom + 0 * stride_coords_dim,
            grad0,
        )
        tl.atomic_add(
            grad_coords_ptr + i * stride_coords_atom + 1 * stride_coords_dim,
            grad1,
        )
        tl.atomic_add(
            grad_coords_ptr + i * stride_coords_atom + 2 * stride_coords_dim,
            grad2,
        )

        tl.atomic_add(
            grad_coords_ptr + j * stride_coords_atom + 0 * stride_coords_dim,
            -grad0,
        )
        tl.atomic_add(
            grad_coords_ptr + j * stride_coords_atom + 1 * stride_coords_dim,
            -grad1,
        )
        tl.atomic_add(
            grad_coords_ptr + j * stride_coords_atom + 2 * stride_coords_dim,
            -grad2,
        )

    class _NonbondedEnergyFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            coords: Tensor,
            index: Tensor,
            vdw_min: Tensor,
            vdw_depth: Tensor,
            thresh_sq: Tensor,
        ) -> Tensor:
            coords_contig = coords.contiguous()
            index_contig = index.contiguous()
            vdw_min = vdw_min.contiguous()
            vdw_depth = vdw_depth.contiguous()
            thresh_sq = thresh_sq.contiguous()

            n_pairs = index_contig.shape[0]
            if n_pairs == 0:
                ctx.save_for_backward(
                    coords_contig,
                    index_contig,
                    vdw_min,
                    vdw_depth,
                    thresh_sq,
                )
                ctx.n_pairs = 0
                ctx.stride_coords_atom = coords_contig.stride(0)
                ctx.stride_coords_dim = coords_contig.stride(1)
                ctx.stride_index_pair = index_contig.stride(0)
                ctx.stride_index_component = index_contig.stride(1)
                return torch.zeros((), device=coords.device, dtype=coords.dtype)

            output = torch.empty(
                (n_pairs,), device=coords.device, dtype=coords.dtype
            )

            grid: Callable[[dict], tuple[int, ...]] = lambda meta: (
                triton.cdiv(n_pairs, meta["BLOCK_SIZE"]),
            )
            _nonbonded_forward_kernel[grid](
                coords_contig,
                index_contig,
                vdw_min,
                vdw_depth,
                thresh_sq,
                output,
                n_pairs,
                coords_contig.stride(0),
                coords_contig.stride(1),
                index_contig.stride(0),
                index_contig.stride(1),
                BLOCK_SIZE=_BLOCK_SIZE,
            )

            ctx.save_for_backward(
                coords_contig,
                index_contig,
                vdw_min,
                vdw_depth,
                thresh_sq,
            )
            ctx.n_pairs = n_pairs
            ctx.stride_coords_atom = coords_contig.stride(0)
            ctx.stride_coords_dim = coords_contig.stride(1)
            ctx.stride_index_pair = index_contig.stride(0)
            ctx.stride_index_component = index_contig.stride(1)

            return output.sum()

        @staticmethod
        def backward(ctx, grad_output: Tensor):
            (coords, index, vdw_min, vdw_depth, thresh_sq) = ctx.saved_tensors
            grad_coords = torch.zeros_like(coords)

            if ctx.n_pairs:
                grid: Callable[[dict], tuple[int, ...]] = lambda meta: (
                    triton.cdiv(ctx.n_pairs, meta["BLOCK_SIZE"]),
                )
                _nonbonded_backward_kernel[grid](
                    coords,
                    index,
                    vdw_min,
                    vdw_depth,
                    thresh_sq,
                    grad_coords,
                    ctx.n_pairs,
                    ctx.stride_coords_atom,
                    ctx.stride_coords_dim,
                    ctx.stride_index_pair,
                    ctx.stride_index_component,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )

            grad_coords.mul_(grad_output)
            return grad_coords, None, None, None, None

    def triton_nonbonded_energy(
        coords: Tensor,
        index: Tensor,
        vdw_min: Tensor,
        vdw_depth: Tensor,
        thresh_sq: Tensor,
    ) -> Tensor:
        """Evaluate the Lennard-Jones energy using a Triton kernel."""

        if coords.dim() == 2:
            return _NonbondedEnergyFunction.apply(
                coords, index, vdw_min, vdw_depth, thresh_sq
            )

        energies = []
        for b in range(coords.shape[0]):
            energy = _NonbondedEnergyFunction.apply(
                coords[b], index, vdw_min, vdw_depth, thresh_sq
            )
            energies.append(energy)
        return torch.stack(energies, dim=0)

else:

    def triton_nonbonded_energy(
        coords: Tensor,
        index: Tensor,
        vdw_min: Tensor,
        vdw_depth: Tensor,
        thresh_sq: Tensor,
    ) -> Tensor:
        raise RuntimeError("Triton backend is not available.")

