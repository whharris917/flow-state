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


# =============================================================================
# R2 emergent-phenomena molecules (amphiphiles + chain polymer)
# =============================================================================
#
# These coarse-grained molecules pair with the R2 PRESET_MATERIALS additions
# (Polar / Nonpolar / Heavy / LightGas) and the seeded cross-ε overrides
# (Sketch._seed_default_lj_overrides). In a Polar-dominant solvent they
# self-organise:
#   - Surfactant: heads stay solvent-exposed, tails cluster → micelles
#   - Lipid: two-tail amphiphile → bilayer-like aggregates / 2D vesicle rings
#   - Polymer: long Nonpolar chain → coil dynamics; collapses in Polar solvent
# Bond k's are stiff enough to resist thermal motion at typical T~0.5–1.0;
# chain angles at 180° (straight) give a mild persistence length without
# being so stiff the chain can't curl.


def make_surfactant(name="Surfactant", head_material="Polar",
                    tail_material="Nonpolar", n_tail=4,
                    r_eq=1.0, k=300.0, k_angle=30.0):
    """Linear amphiphile: 1 Polar head + n_tail Nonpolar atoms in a chain.

    In a Polar solvent the heads stay solvent-exposed (favourable cross-ε
    with the solvent ≈ 1.0 via L-B on like-Polar) while the tails seek each
    other out (favourable Nonpolar-Nonpolar) and avoid the solvent (Polar-
    Nonpolar override ε=0.25). Aggregation produces micelles.

    Geometry: atoms laid out along the local x-axis at spacing r_eq.
    Head at the left end, tails to the right. Mild angle springs (k_angle
    small) along the chain give a persistence length without rigid rod
    behaviour — useful so the surfactant can curve to fit the micelle's
    surface.
    """
    import math
    n_atoms = 1 + n_tail
    atoms = []
    # Head at left end, then n_tail tails marching right. Centre at x=0.
    for i in range(n_atoms):
        x = (i - 0.5 * (n_atoms - 1)) * r_eq
        material = head_material if i == 0 else tail_material
        atoms.append(MoleculeAtom(x=x, y=0.0, material_name=material))
    # Series bonds along the chain.
    bonds = [
        MoleculeBond(atom_a=i, atom_b=i + 1, k=k, r_eq=r_eq)
        for i in range(n_atoms - 1)
    ]
    # 180° chain angles at every interior atom — keeps the chain extended-ish.
    angles = [
        MoleculeAngle(atom_a=i - 1, atom_b=i, atom_c=i + 1,
                      k=k_angle, theta_eq=math.pi)
        for i in range(1, n_atoms - 1)
    ]
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)


def make_lipid(name="Lipid", head_material="Polar", tail_material="Nonpolar",
               r_eq=1.0, k=300.0, k_angle=40.0,
               tail_branch_angle_deg=60.0):
    """Two-tail amphiphile: 1 head + 1 neck (both Polar) + 2 Nonpolar chains
    of 2 atoms each branching from the neck. 6 atoms, 5 bonds, 5 angles.

    Structure (local coords, head at top, tails splayed downward):
        idx 0: head    (Polar)   at (0, +r_eq)
        idx 1: neck    (Polar)   at (0,  0.0)
        idx 2: tail1a  (Nonpolar) branching down-left from neck
        idx 3: tail1b  (Nonpolar) further along the down-left chain
        idx 4: tail2a  (Nonpolar) branching down-right from neck
        idx 5: tail2b  (Nonpolar) further along the down-right chain

    The two tails splay apart at the neck by `tail_branch_angle_deg`
    (default 60°, total opening 120°). With the R2 cross-ε defaults this
    self-assembles into bilayer-like structures where the Polar heads
    bookend the Nonpolar interior. In 2D the bilayer looks like two
    parallel ribbons of heads facing the solvent and tails packed in
    between, or curls into a "2D vesicle" ring.

    Why split head and neck (both Polar): the neck atom acts as a flex
    pivot — the lipid can hinge at the head–neck bond, which improves
    packing into curved bilayer surfaces vs a rigid 3-armed star.
    """
    import math
    half_branch = math.radians(tail_branch_angle_deg) / 2.0  # half-opening from -y
    cos_b = math.cos(half_branch)
    sin_b = math.sin(half_branch)
    # tail1a sits at (-sin_b·r_eq, -cos_b·r_eq) from the neck.
    t1a_x = -sin_b * r_eq
    t1a_y = -cos_b * r_eq
    t2a_x = +sin_b * r_eq
    t2a_y = -cos_b * r_eq
    # tail1b/tail2b continue in the SAME direction as the branch (down and
    # outward) so the chain straightens.
    t1b_x = 2.0 * t1a_x
    t1b_y = 2.0 * t1a_y
    t2b_x = 2.0 * t2a_x
    t2b_y = 2.0 * t2a_y
    atoms = [
        MoleculeAtom(x=0.0, y=+r_eq, material_name=head_material),         # 0 head
        MoleculeAtom(x=0.0, y=0.0,   material_name=head_material),         # 1 neck
        MoleculeAtom(x=t1a_x, y=t1a_y, material_name=tail_material),       # 2 tail1a
        MoleculeAtom(x=t1b_x, y=t1b_y, material_name=tail_material),       # 3 tail1b
        MoleculeAtom(x=t2a_x, y=t2a_y, material_name=tail_material),       # 4 tail2a
        MoleculeAtom(x=t2b_x, y=t2b_y, material_name=tail_material),       # 5 tail2b
    ]
    bonds = [
        MoleculeBond(atom_a=0, atom_b=1, k=k, r_eq=r_eq),  # head–neck
        MoleculeBond(atom_a=1, atom_b=2, k=k, r_eq=r_eq),  # neck–tail1a
        MoleculeBond(atom_a=2, atom_b=3, k=k, r_eq=r_eq),  # tail1a–tail1b
        MoleculeBond(atom_a=1, atom_b=4, k=k, r_eq=r_eq),  # neck–tail2a
        MoleculeBond(atom_a=4, atom_b=5, k=k, r_eq=r_eq),  # tail2a–tail2b
    ]
    # Angles at neck (1): hold the head-up + tails-down + tail-branch geometry.
    # Two straightening angles along the tail chains (180°).
    head_to_tail_angle = math.pi - half_branch  # head→neck→tail1a opening
    angles = [
        # head–neck–tail1a: 180° - half_branch (head up, tail1a down-left)
        MoleculeAngle(atom_a=0, atom_b=1, atom_c=2,
                      k=k_angle, theta_eq=head_to_tail_angle),
        # head–neck–tail2a: same opening on the other side
        MoleculeAngle(atom_a=0, atom_b=1, atom_c=4,
                      k=k_angle, theta_eq=head_to_tail_angle),
        # tail1a–neck–tail2a: 2·half_branch (the tail-branch angle)
        MoleculeAngle(atom_a=2, atom_b=1, atom_c=4,
                      k=k_angle, theta_eq=2.0 * half_branch),
        # neck–tail1a–tail1b: 180° (chain extension)
        MoleculeAngle(atom_a=1, atom_b=2, atom_c=3,
                      k=k_angle, theta_eq=math.pi),
        # neck–tail2a–tail2b: 180° (chain extension)
        MoleculeAngle(atom_a=1, atom_b=4, atom_c=5,
                      k=k_angle, theta_eq=math.pi),
    ]
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)


def make_polymer(name="Polymer", material="Nonpolar", n=15,
                 r_eq=1.0, k=300.0, k_angle=20.0):
    """Long bonded chain of `n` atoms (all the same material) — useful for
    Rouse / reptation dynamics demos and as a high-molecular-weight contrast
    to the diatomic / small-molecule entries in the palette.

    Default material is Nonpolar so the polymer collapses in a Polar solvent
    (a "bad solvent" — the chain coils to minimise solvent contact, mimicking
    real polymer θ-conditions). Override `material` to "Polar" for a soluble
    chain that swells into the solvent.

    Atoms are laid out along the local x-axis at spacing r_eq, centred on
    the origin. Bond k is stiff (300); chain angles are gentle (k_angle=20)
    to let the chain coil under bad-solvent conditions without being
    floppy enough to self-overlap.
    """
    import math
    atoms = [
        MoleculeAtom(x=(i - 0.5 * (n - 1)) * r_eq, y=0.0,
                     material_name=material)
        for i in range(n)
    ]
    bonds = [
        MoleculeBond(atom_a=i, atom_b=i + 1, k=k, r_eq=r_eq)
        for i in range(n - 1)
    ]
    angles = [
        MoleculeAngle(atom_a=i - 1, atom_b=i, atom_c=i + 1,
                      k=k_angle, theta_eq=math.pi)
        for i in range(1, n - 1)
    ]
    return MoleculeTemplate(name=name, atoms=atoms, bonds=bonds, angles=angles)
