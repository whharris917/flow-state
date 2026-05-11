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
class MoleculeAngle:
    """One three-body angle spring inside a molecule template.

    atom_b is the apex (vertex); atom_a and atom_c are the legs. Energy is
    U(θ) = k*(θ - theta_eq)² where θ is the angle between vectors b→a and
    b→c. theta_eq is in radians (0 to π).
    """
    atom_a: int
    atom_b: int
    atom_c: int
    k: float
    theta_eq: float

    def to_dict(self):
        return {
            'atom_a': int(self.atom_a),
            'atom_b': int(self.atom_b),
            'atom_c': int(self.atom_c),
            'k': float(self.k),
            'theta_eq': float(self.theta_eq),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            atom_a=int(data['atom_a']),
            atom_b=int(data['atom_b']),
            atom_c=int(data['atom_c']),
            k=float(data['k']),
            theta_eq=float(data['theta_eq']),
        )


@dataclass
class MoleculeTemplate:
    """A named, re-usable cluster of atoms + bonds + angles."""
    name: str
    atoms: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)

    def to_dict(self):
        return {
            'name': self.name,
            'atoms': [a.to_dict() for a in self.atoms],
            'bonds': [b.to_dict() for b in self.bonds],
            'angles': [a.to_dict() for a in self.angles],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=str(data['name']),
            atoms=[MoleculeAtom.from_dict(a) for a in data.get('atoms', [])],
            bonds=[MoleculeBond.from_dict(b) for b in data.get('bonds', [])],
            angles=[MoleculeAngle.from_dict(a) for a in data.get('angles', [])],
        )

    def validate(self):
        """Return True iff every bond's and angle's atom indices are valid.
        Templates with broken indices are rejected at placement time."""
        n = len(self.atoms)
        for b in self.bonds:
            if not (0 <= b.atom_a < n):
                return False
            if not (0 <= b.atom_b < n):
                return False
            if b.atom_a == b.atom_b:
                return False
        for ang in self.angles:
            if not (0 <= ang.atom_a < n):
                return False
            if not (0 <= ang.atom_b < n):
                return False
            if not (0 <= ang.atom_c < n):
                return False
            if ang.atom_a == ang.atom_b or ang.atom_b == ang.atom_c or ang.atom_a == ang.atom_c:
                return False
        return True

    def auto_generate_angles(self, k=50.0):
        """Infer angle springs from current bond topology + atom positions.

        For every pair of bonds sharing a common atom (the apex), an angle
        constraint is added with theta_eq taken from the *current* geometry
        of the template's atom positions. Overwrites any existing angles.

        Args:
            k: Angular stiffness for every generated angle.

        Returns:
            Number of angles generated.
        """
        import math
        # Build adjacency: apex_atom_idx → list of (partner_idx, bond_idx)
        adjacency = {}
        for b_idx, b in enumerate(self.bonds):
            adjacency.setdefault(b.atom_a, []).append(b.atom_b)
            adjacency.setdefault(b.atom_b, []).append(b.atom_a)

        self.angles = []
        for apex, partners in adjacency.items():
            # Every unordered pair of partners forms an angle at this apex
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    a_idx = partners[i]
                    c_idx = partners[j]
                    # Compute theta_eq from current positions
                    a = self.atoms[a_idx]
                    b = self.atoms[apex]
                    c = self.atoms[c_idx]
                    dxba = a.x - b.x
                    dyba = a.y - b.y
                    dxbc = c.x - b.x
                    dybc = c.y - b.y
                    r_ba = math.hypot(dxba, dyba)
                    r_bc = math.hypot(dxbc, dybc)
                    if r_ba < 1e-9 or r_bc < 1e-9:
                        continue
                    cos_th = (dxba * dxbc + dyba * dybc) / (r_ba * r_bc)
                    cos_th = max(-1.0, min(1.0, cos_th))
                    theta = math.acos(cos_th)
                    self.angles.append(MoleculeAngle(
                        atom_a=a_idx, atom_b=apex, atom_c=c_idx,
                        k=float(k), theta_eq=float(theta),
                    ))
        return len(self.angles)


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
               oh_r_eq=1.0, k=400.0, k_angle=80.0):
    """V-shaped 3-atom "water-like" molecule.

    Uses Mercury (heavy/dense) for the central "O" and Water (light) for the
    two "H" atoms — gives a visible mass contrast at placement time without
    introducing brand-new material names. H-O-H angle is ~104.5° (the real
    water angle); orientation places the O at the local origin and the two
    H atoms below the x-axis, symmetric about y. The angle is held by an
    explicit MoleculeAngle so the geometry survives thermal motion instead
    of collapsing to linear under repulsion.
    """
    import math
    theta_eq = math.radians(104.5)
    half_angle = theta_eq / 2.0
    h_x = oh_r_eq * math.sin(half_angle)
    h_y = -oh_r_eq * math.cos(half_angle)
    return MoleculeTemplate(
        name=name,
        atoms=[
            MoleculeAtom(x=0.0, y=0.0, material_name=o_material),       # central O
            MoleculeAtom(x=-h_x, y=h_y, material_name=h_material),      # H1
            MoleculeAtom(x=+h_x, y=h_y, material_name=h_material),      # H2
        ],
        bonds=[
            MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=oh_r_eq),
            MoleculeBond(atom_a=0, atom_b=2, k=k, r_eq=oh_r_eq),
        ],
        angles=[
            MoleculeAngle(atom_a=1, atom_b=0, atom_c=2,
                          k=k_angle, theta_eq=theta_eq),
        ],
    )


def make_co2(name="CO2", c_material="Honey", o_material="Mercury",
             co_r_eq=1.2, k=500.0, k_angle=120.0):
    """Linear 3-atom CO₂-like molecule. C in the centre, O on each side at
    180°. The angle spring resists any deflection from linearity.
    """
    import math
    return MoleculeTemplate(
        name=name,
        atoms=[
            MoleculeAtom(x=-co_r_eq, y=0.0, material_name=o_material),  # O1
            MoleculeAtom(x=0.0, y=0.0, material_name=c_material),       # central C
            MoleculeAtom(x=+co_r_eq, y=0.0, material_name=o_material),  # O2
        ],
        bonds=[
            MoleculeBond(atom_a=1, atom_b=0, k=k, r_eq=co_r_eq),
            MoleculeBond(atom_a=1, atom_b=2, k=k, r_eq=co_r_eq),
        ],
        angles=[
            MoleculeAngle(atom_a=0, atom_b=1, atom_c=2,
                          k=k_angle, theta_eq=math.pi),
        ],
    )


def make_ammonia(name="Ammonia", n_material="Mercury", h_material="Water",
                 nh_r_eq=1.0, k=350.0, k_angle=70.0):
    """Trigonal-planar (in 2D) 4-atom NH₃-like molecule. N at centre with
    three H atoms at 120° spacing. Real NH₃ is pyramidal in 3D; flattening
    to 2D gives an exactly-planar trigonal pattern.
    """
    import math
    theta_eq = math.radians(120.0)
    atoms = [MoleculeAtom(x=0.0, y=0.0, material_name=n_material)]
    for i in range(3):
        a = -math.pi / 2 + i * theta_eq  # spread the 3 H atoms around the N
        atoms.append(MoleculeAtom(
            x=nh_r_eq * math.cos(a), y=nh_r_eq * math.sin(a),
            material_name=h_material,
        ))
    bonds = [
        MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=nh_r_eq),
        MoleculeBond(atom_a=0, atom_b=2, k=k, r_eq=nh_r_eq),
        MoleculeBond(atom_a=0, atom_b=3, k=k, r_eq=nh_r_eq),
    ]
    angles = [
        MoleculeAngle(atom_a=1, atom_b=0, atom_c=2,
                      k=k_angle, theta_eq=theta_eq),
        MoleculeAngle(atom_a=2, atom_b=0, atom_c=3,
                      k=k_angle, theta_eq=theta_eq),
        MoleculeAngle(atom_a=3, atom_b=0, atom_c=1,
                      k=k_angle, theta_eq=theta_eq),
    ]
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)


def make_methane(name="Methane", c_material="Honey", h_material="Water",
                 ch_r_eq=1.0, k=350.0, k_angle=70.0):
    """5-atom CH₄-like molecule, 2D-planar with 4 hydrogens at 90° spacing.
    True methane is tetrahedral (109.5°) in 3D — the 2D collapse gives a
    plus-shape with one H above, below, left, and right of the central C.
    """
    import math
    theta_eq = math.radians(90.0)
    atoms = [MoleculeAtom(x=0.0, y=0.0, material_name=c_material)]
    for i in range(4):
        a = i * (math.pi / 2)
        atoms.append(MoleculeAtom(
            x=ch_r_eq * math.cos(a), y=ch_r_eq * math.sin(a),
            material_name=h_material,
        ))
    bonds = [
        MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=ch_r_eq),
        MoleculeBond(atom_a=0, atom_b=2, k=k, r_eq=ch_r_eq),
        MoleculeBond(atom_a=0, atom_b=3, k=k, r_eq=ch_r_eq),
        MoleculeBond(atom_a=0, atom_b=4, k=k, r_eq=ch_r_eq),
    ]
    # Adjacent H-C-H angles (4 of them around the central C)
    angles = []
    for i in range(4):
        a = i + 1
        c = ((i + 1) % 4) + 1
        angles.append(MoleculeAngle(
            atom_a=a, atom_b=0, atom_c=c,
            k=k_angle, theta_eq=theta_eq,
        ))
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)


def make_benzene(name="Benzene", c_material="Honey",
                 cc_r_eq=1.4, k=400.0, k_angle=100.0):
    """6-atom benzene-like ring (regular hexagon, all atoms same material).
    Each adjacent pair is bonded; each interior angle at every vertex is
    constrained to 120° via an angle spring. The ring's rigidity comes
    from the combination of 6 bonds + 6 angles.
    """
    import math
    theta_eq = math.radians(120.0)
    # Place 6 atoms on a regular hexagon of radius R such that adjacent
    # atoms are cc_r_eq apart. For a regular hexagon, side = R.
    R = cc_r_eq
    atoms = []
    for i in range(6):
        a = i * (math.pi / 3) - math.pi / 2
        atoms.append(MoleculeAtom(
            x=R * math.cos(a), y=R * math.sin(a),
            material_name=c_material,
        ))
    bonds = []
    for i in range(6):
        bonds.append(MoleculeBond(
            atom_a=i, atom_b=(i + 1) % 6,
            k=k, r_eq=cc_r_eq,
        ))
    angles = []
    for i in range(6):
        # Angle at vertex i: legs (i-1) and (i+1), apex i
        a_idx = (i - 1) % 6
        c_idx = (i + 1) % 6
        angles.append(MoleculeAngle(
            atom_a=a_idx, atom_b=i, atom_c=c_idx,
            k=k_angle, theta_eq=theta_eq,
        ))
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)
