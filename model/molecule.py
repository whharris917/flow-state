"""
Molecule templates — re-usable particle clusters with intramolecular bonds.

A MoleculeTemplate is a Sketch-side blueprint. Each template has:
- N MoleculeAtom entries (local coords + material name per atom)
- M MoleculeBond entries (atom-index pairs + per-bond k / r_eq)

Placing a template via AddMoleculeCommand spawns N atoms and M bonds into
the Simulation. Templates themselves never enter the Simulation — they are
authored in the builder, stored on Sketch.molecules, and instanced per
placement.

The local coordinate system has the molecule's "centre" at (0, 0). Atom
positions are stored in that local frame; AddMoleculeCommand transforms
them into world coordinates at placement time (with optional rotation).
"""

from dataclasses import dataclass, field


@dataclass
class MoleculeAtom:
    """One atom inside a molecule template. Local coords + material name."""
    x: float
    y: float
    material_name: str

    def to_dict(self):
        return {
            'x': float(self.x),
            'y': float(self.y),
            'material_name': self.material_name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            x=float(data['x']),
            y=float(data['y']),
            material_name=str(data['material_name']),
        )


@dataclass
class MoleculeBond:
    """One harmonic bond inside a molecule template.

    atom_a and atom_b index into the parent MoleculeTemplate.atoms list.
    k and r_eq are the spring stiffness and equilibrium length applied to
    every instance of this molecule at placement time.
    """
    atom_a: int
    atom_b: int
    k: float
    r_eq: float

    def to_dict(self):
        return {
            'atom_a': int(self.atom_a),
            'atom_b': int(self.atom_b),
            'k': float(self.k),
            'r_eq': float(self.r_eq),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            atom_a=int(data['atom_a']),
            atom_b=int(data['atom_b']),
            k=float(data['k']),
            r_eq=float(data['r_eq']),
        )


@dataclass
class MoleculeTemplate:
    """A named, re-usable cluster of atoms + bonds."""
    name: str
    atoms: list = field(default_factory=list)
    bonds: list = field(default_factory=list)

    def to_dict(self):
        return {
            'name': self.name,
            'atoms': [a.to_dict() for a in self.atoms],
            'bonds': [b.to_dict() for b in self.bonds],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=str(data['name']),
            atoms=[MoleculeAtom.from_dict(a) for a in data.get('atoms', [])],
            bonds=[MoleculeBond.from_dict(b) for b in data.get('bonds', [])],
        )

    def validate(self):
        """Return True iff every bond's atom indices reference valid atoms.
        Templates with broken bond indices are rejected at placement time."""
        n = len(self.atoms)
        for b in self.bonds:
            if not (0 <= b.atom_a < n):
                return False
            if not (0 <= b.atom_b < n):
                return False
            if b.atom_a == b.atom_b:
                return False
        return True


# =============================================================================
# Convenience builders for seeded "starter" molecules
# =============================================================================

def make_diatom(name="Diatom", material_name="Water", r_eq=1.0, k=200.0):
    """Two atoms of the same material bonded along the local x-axis at r_eq.

    The simplest possible molecule; useful as a starter palette entry and a
    test stimulus. Both atoms share the same material so colour and physics
    parameters are uniform.
    """
    return MoleculeTemplate(
        name=name,
        atoms=[
            MoleculeAtom(x=-0.5 * r_eq, y=0.0, material_name=material_name),
            MoleculeAtom(x=+0.5 * r_eq, y=0.0, material_name=material_name),
        ],
        bonds=[MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=r_eq)],
    )


def make_water(name="Water-mol", o_material="Mercury", h_material="Water",
               oh_r_eq=1.0, k=400.0):
    """V-shaped 3-atom "water-like" molecule.

    Uses Mercury (heavy/dense) for the central "O" and Water (light) for the
    two "H" atoms — gives a visible mass contrast at placement time without
    introducing brand-new material names. H-O-H angle is ~104.5° (the real
    water angle); orientation places the O at the local origin and the two
    H atoms below the x-axis, symmetric about y.

    Renamed from the bare "Water" so it doesn't collide with the Water
    material when serialized in Sketch.molecules.
    """
    import math
    half_angle = math.radians(52.25)
    h_x = oh_r_eq * math.sin(half_angle)
    h_y = -oh_r_eq * math.cos(half_angle)
    return MoleculeTemplate(
        name=name,
        atoms=[
            MoleculeAtom(x=0.0, y=0.0, material_name=o_material),       # central
            MoleculeAtom(x=-h_x, y=h_y, material_name=h_material),      # H1
            MoleculeAtom(x=+h_x, y=h_y, material_name=h_material),      # H2
        ],
        bonds=[
            MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=oh_r_eq),
            MoleculeBond(atom_a=0, atom_b=2, k=k, r_eq=oh_r_eq),
        ],
    )
