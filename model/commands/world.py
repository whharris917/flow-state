"""World-Property Commands — operate on Simulation-level state (not Sketch).

These commands wrap operations that mutate world / physics state (e.g.,
world_size resize) so they participate in the same Command queue as Sketch
mutations. Without this, AppController.action_undo prefers `scene.can_undo()`
(the CAD queue) over `sim.undo()` (the physics snapshot stack), so a CAD
command can hide a destructive resize behind it — a single Ctrl+Z then
undoes the wrong thing.

Architecturally these are still Vault state per CLAUDE.md §4.1 — anything
saved into a `.scn` file (and `world_size` is) must flow through the Command
queue.
"""

import numpy as np

from core.command_base import Command


class ResizeWorldCommand(Command):
    """Resize the simulation world with snapshot-based undo.

    `Simulation.resize_world(new_size)` performs a full physics reset (clears
    particles + restores physics defaults). The undo path captures the entire
    pre-resize physics state plus world_size and restores it on undo. After
    restoration we run `scene.rebuild()` to re-emit the static/tethered atoms
    from the current sketch geometry, which the captured-state restore wiped
    along with the rest.

    Notes:
        * We do NOT inherit `changes_topology=True`. resize-and-rebuild is
          performed eagerly inside `execute()` and `undo()`; setting the flag
          would just trigger a redundant rebuild on the next `Scene.update()`.
        * `Simulation.resize_world` snapshots internally (its own undo stack).
          That snapshot is harmless — this command's undo path uses its own
          captured state and never calls `sim.undo()`.
    """

    def __init__(self, scene, new_size, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.scene = scene
        self.new_size = float(new_size)
        self.captured_state = None
        self.description = f"Resize World"

    def execute(self) -> bool:
        sim = self.scene.simulation
        # Capture the full physics state before the destructive reset
        self.captured_state = {
            'count': sim.count,
            'pos_x': np.copy(sim.pos_x[:sim.count]),
            'pos_y': np.copy(sim.pos_y[:sim.count]),
            'vel_x': np.copy(sim.vel_x[:sim.count]),
            'vel_y': np.copy(sim.vel_y[:sim.count]),
            'is_static': np.copy(sim.is_static[:sim.count]),
            'atom_sigma': np.copy(sim.atom_sigma[:sim.count]),
            'atom_eps_sqrt': np.copy(sim.atom_eps_sqrt[:sim.count]),
            'atom_color': np.copy(sim.atom_color[:sim.count]),
            'world_size': sim.world_size,
        }
        sim.resize_world(self.new_size)
        # Restore Compiler-emitted atoms — sim.reset() wiped them
        self.scene.rebuild()
        return True

    def undo(self):
        if self.captured_state is None:
            return
        sim = self.scene.simulation
        sim._restore_physics_state(self.captured_state)
        # Re-emit static/tethered atoms from the (unchanged) sketch geometry
        self.scene.rebuild()
