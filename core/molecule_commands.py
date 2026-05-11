"""
Molecule Commands — Undo/Redo Support for Molecule Placement

AddMoleculeCommand instances a MoleculeTemplate at a world position into
N atoms and M bonds in the Simulation. Undo truncates back to the pre-
placement atom_count and bond_count.

Undo limitation (documented):
    Because atom/bond storage is index-based and compact_arrays shifts
    indices on removal, "undo a molecule" is implemented as a count
    truncation rather than per-atom selective removal. If new atoms or
    bonds were added AFTER this molecule placement (e.g., via the brush,
    which has its own physics-snapshot undo stack — see
    AppController.action_undo, which prefers scene.can_undo before
    sim.undo), those would be lost on this command's undo. This matches
    the existing limitation in sim.undo (which restores a full pre-stroke
    snapshot, also losing anything added in between).

    The truncation is safe in the common case: place molecule → Ctrl+Z
    immediately, with no intervening atom-adding operations.
"""

import math

from core.commands import Command


class AddMoleculeCommand(Command):
    """Place a MoleculeTemplate at world (cx, cy) with optional rotation.

    Each MoleculeAtom's local (x, y) is rotated by `rotation` radians and
    translated by (cx, cy) to give the spawn world position. Material
    parameters (sigma, epsilon, mass, color) are resolved via
    sketch.get_material(atom.material_name) at execute time so the scene's
    current material palette wins over whatever the template thinks.

    Each MoleculeBond uses its own k and r_eq verbatim (the builder is
    responsible for seeding sensible defaults via Sketch.get_bond_default).
    """

    changes_topology = False  # Particle-level placement, not Sketch topology

    def __init__(self, scene, template, world_pos, rotation=0.0,
                 historize=True, supersede=False):
        super().__init__(historize=historize, supersede=supersede)
        self.scene = scene
        self.template = template
        self.world_pos = (float(world_pos[0]), float(world_pos[1]))
        self.rotation = float(rotation)
        # Captured at execute(); used by undo() to truncate back to pre-state.
        self._pre_atom_count = None
        self._pre_bond_count = None
        self.description = f"Add Molecule '{template.name}'"

    def execute(self) -> bool:
        sim = self.scene.simulation
        sketch = self.scene.sketch

        if not self.template.validate():
            return False

        # Capture pre-state for undo (truncation-based — see module docstring).
        self._pre_atom_count = sim.count
        self._pre_bond_count = sim.bond_count

        cx, cy = self.world_pos
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        atom_indices = []
        for ma in self.template.atoms:
            mat = sketch.get_material(ma.material_name)
            # Rotate local → world. Standard 2D rotation matrix:
            #   wx = cos*x - sin*y; wy = sin*x + cos*y
            wx = cx + cos_r * ma.x - sin_r * ma.y
            wy = cy + sin_r * ma.x + cos_r * ma.y
            idx = sim._add_particle(
                wx, wy, vx=0.0, vy=0.0, is_static=0,
                sigma=getattr(mat, 'sigma', None),
                epsilon=getattr(mat, 'epsilon', None),
                mass=getattr(mat, 'mass', None),
                color=getattr(mat, 'color', (50, 150, 255)),
            )
            atom_indices.append(idx)

        for mb in self.template.bonds:
            i = atom_indices[mb.atom_a]
            j = atom_indices[mb.atom_b]
            sim.add_bond(i, j, k=mb.k, r_eq=mb.r_eq)

        sim.rebuild_next = True
        return True

    def undo(self):
        sim = self.scene.simulation
        if self._pre_atom_count is None:
            return
        # Truncate atom and bond counts. This is correct when no other
        # particle-adding operation has happened since execute(); see
        # module docstring for the limitation.
        sim.count = max(0, self._pre_atom_count)
        sim.bond_count = max(0, self._pre_bond_count)
        sim.rebuild_next = True

    def redo(self):
        # Re-execute. Bond and atom counts will repopulate from the same
        # template at the same world_pos / rotation.
        self.execute()
