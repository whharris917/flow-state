"""
Experimental utilities for the benchmark suite.

Used to evaluate engine-level perf changes externally — without touching
production code — so we can measure first and justify the engine change with
data, not the other way around.

Currently houses:

- `external_spatial_sort(sim)`: re-orders all per-atom arrays in `sim` by
  spatial cell key (cy * n_cells + cx). Mirrors what
  `engine.physics_core.spatial_sort` does for the basic eight arrays, but
  also permutes the arrays the production kernel touches that the existing
  `spatial_sort` doesn't know about (last_x/y, atom_color, tether_*,
  joint_ids). Does NOT modify production engine code.

  The neighbour list (pair_i, pair_j, pair_count, nbr_start, nbr_idx) refers
  to atoms by index; we invalidate it via `sim.rebuild_next = True`. The next
  Simulation.step() call rebuilds it from scratch on the sorted layout.
"""

from __future__ import annotations

import numpy as np


# Per-atom array attribute names on Simulation. Each gets the same permutation
# applied. 1D arrays go in `_PERATOM_1D`; 2D arrays in `_PERATOM_2D`.
_PERATOM_1D = (
    "pos_x", "pos_y",
    "vel_x", "vel_y",
    "force_x", "force_y",
    "last_x", "last_y",
    "is_static",
    "atom_sigma", "atom_eps_sqrt",
    "tether_entity_idx", "tether_stiffness",
    "joint_ids",
)
_PERATOM_2D = (
    "atom_color",
    "tether_local_pos",
)


def external_spatial_sort(sim) -> int:
    """Re-order all per-atom arrays of `sim` by spatial cell key.

    Returns the number of atoms sorted. Sets `sim.rebuild_next = True` so the
    next step() rebuilds the neighbour list against the new layout.

    Implemented in numpy so we don't have to touch production engine code to
    test whether spatial sorting is worthwhile.
    """
    N = sim.count
    if N == 0:
        return 0

    cell_size = sim.cell_size
    inv_cell = 1.0 / cell_size
    n_cells = int(sim.world_size // cell_size) + 1

    cx = (sim.pos_x[:N] * inv_cell).astype(np.int32)
    cy = (sim.pos_y[:N] * inv_cell).astype(np.int32)
    keys = cy * n_cells + cx
    perm = np.argsort(keys, kind="stable")

    for name in _PERATOM_1D:
        arr = getattr(sim, name)
        arr[:N] = arr[:N][perm]
    for name in _PERATOM_2D:
        arr = getattr(sim, name)
        arr[:N] = arr[:N][perm]

    sim.rebuild_next = True
    return N
