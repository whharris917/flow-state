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
import random

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
        self._pre_angle_count = None
        self.description = f"Add Molecule '{template.name}'"

    def execute(self) -> bool:
        sim = self.scene.simulation
        sketch = self.scene.sketch

        if not self.template.validate():
            return False

        # Capture pre-state for undo (truncation-based — see module docstring).
        self._pre_atom_count = sim.count
        self._pre_bond_count = sim.bond_count
        self._pre_angle_count = sim.angle_count

        cx, cy = self.world_pos
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        # Sample a Maxwell-Boltzmann centre-of-mass velocity for the molecule.
        # The thermostat is a multiplicative Berendsen rescaler — it cannot
        # heat from absolute zero (early-out at current_T <= 1e-6). Placing
        # molecules at LJ + bond equilibrium with zero velocity leaves the
        # net force at zero, so without this kick they sit perfectly still
        # and the thermostat does nothing. Applying the same (vx_cm, vy_cm)
        # to every atom in the molecule gives translational thermal motion
        # without immediately exciting internal vibration; the thermostat
        # then maintains it.
        molecule_mass = 0.0
        for ma in self.template.atoms:
            mat = sketch.get_material(ma.material_name)
            molecule_mass += getattr(mat, 'mass', 1.0)
        target_temp = float(getattr(sim, 'target_temp', 0.0))
        if molecule_mass > 0.0 and target_temp > 0.0:
            std = math.sqrt(target_temp / molecule_mass)
            vx_cm = random.gauss(0.0, std)
            vy_cm = random.gauss(0.0, std)
        else:
            vx_cm = 0.0
            vy_cm = 0.0

        atom_indices = []
        for ma in self.template.atoms:
            mat = sketch.get_material(ma.material_name)
            # Stable material id so cross-pair ε overrides apply to molecule
            # atoms. Templates may carry material names that aren't currently
            # in sketch.materials (e.g. a saved template referencing a
            # since-deleted material); get_material_index returns -1 in that
            # case → kernel falls back to per-atom ε_sqrt for that atom.
            material_id = sketch.get_material_index(ma.material_name)
            # Rotate local → world. Standard 2D rotation matrix:
            #   wx = cos*x - sin*y; wy = sin*x + cos*y
            wx = cx + cos_r * ma.x - sin_r * ma.y
            wy = cy + sin_r * ma.x + cos_r * ma.y
            idx = sim._add_particle(
                wx, wy, vx=vx_cm, vy=vy_cm, is_static=0,
                sigma=getattr(mat, 'sigma', None),
                epsilon=getattr(mat, 'epsilon', None),
                mass=getattr(mat, 'mass', None),
                color=getattr(mat, 'color', (50, 150, 255)),
                material_id=material_id,
            )
            atom_indices.append(idx)

        for mb in self.template.bonds:
            i = atom_indices[mb.atom_a]
            j = atom_indices[mb.atom_b]
            sim.add_bond(i, j, k=mb.k, r_eq=mb.r_eq)

        # Angles use the same template-local → placement-time index remap.
        # The template's MoleculeAngle entries index into template.atoms;
        # we substitute the freshly-assigned simulation indices.
        for ma in getattr(self.template, 'angles', []):
            a = atom_indices[ma.atom_a]
            b = atom_indices[ma.atom_b]
            c = atom_indices[ma.atom_c]
            sim.add_angle(a, b, c, k=ma.k, theta_eq=ma.theta_eq)

        sim.rebuild_next = True
        return True

    def undo(self):
        sim = self.scene.simulation
        if self._pre_atom_count is None:
            return
        # Truncate atom, bond, and angle counts. This is correct when no
        # other particle/bond/angle-adding operation has happened since
        # execute(); see module docstring for the limitation.
        sim.count = max(0, self._pre_atom_count)
        sim.bond_count = max(0, self._pre_bond_count)
        sim.angle_count = max(0, self._pre_angle_count)
        sim.rebuild_next = True

    def redo(self):
        # Re-execute. Bond and atom counts will repopulate from the same
        # template at the same world_pos / rotation.
        self.execute()
