import copy
import numpy as np
import math
from model.geometry import Line, Circle, Point
from model.constraints import create_constraint
from model.solver import Solver
from model.properties import Material, PRESET_MATERIALS
from model.molecule import (
    MoleculeTemplate,
    make_diatom, make_water, make_co2,
    make_ammonia, make_methane, make_benzene,
    make_surfactant, make_lipid, make_polymer,
)

class Sketch:
    """
    The Sketch is the 'Model' or 'Blueprint'.
    It holds Geometries, Constraints, and Materials.
    """
    def __init__(self):
        self.entities = []      # Lines, Circles, Points
        self.constraints = []   # Constraint Data Objects
        self.drivers = []       # Animation Drivers

        # Material Registry - use preset materials as the single source of truth
        # Copy to avoid mutating the global presets
        self.materials = {name: mat.copy() for name, mat in PRESET_MATERIALS.items()}

        # --- Molecule Templates ---
        # Palette of named, re-usable particle clusters. Authored via the
        # Molecule Builder dialog (R3) or seeded with starter entries below.
        # Each placement instances the template into N atoms + M bonds in
        # the Simulation; templates themselves never enter the Simulation.
        self.molecules: dict = {}

        # Per-material-pair bond defaults: frozenset({matA, matB}) → (k, r_eq).
        # Looked up by the builder UI when a bond is drawn so the user starts
        # with a sensible (k, r_eq) for the pair. Empty until populated by
        # the user or by set_bond_default; get_bond_default falls back to
        # sigma-based geometry when a pair has no override.
        self.bond_defaults: dict = {}

        # Per-material-pair LJ ε overrides: frozenset({matA, matB}) → ε_AB.
        # Breaks Berthelot's geometric-mean mixing rule (ε_ij = √(ε_i·ε_j))
        # so cross-species interactions can be tuned independently of the
        # like-species cohesion. This is the load-bearing mechanism for the
        # interesting emergent phenomena (oil-water demixing, surfactant
        # micelles, wetting): set ε_AB < √(ε_AA·ε_BB) to make A and B
        # immiscible. Empty by default → L-B everywhere. σ_ij continues to
        # use Lorentz arithmetic-mean mixing (not overridden in R1).
        self.lj_cross_overrides: dict = {}

        self._seed_default_molecules()
        self._seed_default_lj_overrides()

        # Solver Configuration (runtime toggles for benchmarking)
        self.use_numba = False          # Default to legacy OOP path for safety
        self.solver_iterations = 20     # Default iteration count

        # Interaction Data (mouse drag as a constraint - "User Servo")
        # Format: {'entity_idx': int, 'point_idx': int or None, 'handle_t': float or None, 'target': (x, y)}
        # - entity_idx: Index of entity being dragged
        # - point_idx: Specific point index (for EDIT mode), or None for body drag
        # - handle_t: Parameter t (0.0-1.0) along line where user grabbed, or None
        # - target: World coordinates the user is dragging toward
        self.interaction_data = None

    def _seed_default_molecules(self):
        """Seed the palette with starter templates. Users author additional
        molecules via the Molecule Builder dialog. Each starter ships with
        explicit MoleculeAngle entries so its geometry survives thermal
        motion (angles resist collapse to linear / collapse-onto-each-other)."""
        for tpl in (
            make_diatom(),
            make_water(),
            make_co2(),
            make_ammonia(),
            make_methane(),
            make_benzene(),
            # R2 emergent-phenomena molecules — coarse-grained amphiphiles
            # and a chain polymer. With the R2 cross-ε defaults seeded
            # below, these should self-organise: Surfactant → micelles,
            # Lipid → bilayer-like aggregates, Polymer → conformational
            # dynamics in either solvent.
            make_surfactant(),
            make_lipid(),
            make_polymer(),
        ):
            self.molecules[tpl.name] = tpl

    def _seed_default_lj_overrides(self):
        """Seed Berthelot-breaking cross-pair ε values so the R2 emergent-
        phenomena species (Polar / Nonpolar / Heavy / LightGas) ship with
        recognisable immiscibility relationships out of the box. A fresh
        Sketch + a placed Surfactant in a Polar solvent should immediately
        demonstrate aggregation, with no further user setup. Values follow
        the recipe in the design doc:
          - Polar / Nonpolar  → 0.25 (immiscible, the workhorse "oil-water" pair)
          - Polar / LightGas  → 0.15 (gas barely dissolves in polar liquid)
          - Heavy / LightGas  → 0.20 (gas doesn't dissolve in dense fluid)
          - Nonpolar / Heavy  → 1.5  (oil dissolves heavy nonpolar species)
        Pre-R1 / pre-R2 saves with no overrides recorded restore as empty
        and the user gets pure L-B everywhere — the seed only fires on
        fresh Sketch __init__, not on Sketch.restore."""
        self.set_lj_cross_override("Polar", "Nonpolar", 0.25)
        self.set_lj_cross_override("Polar", "LightGas", 0.15)
        self.set_lj_cross_override("Heavy", "LightGas", 0.20)
        self.set_lj_cross_override("Nonpolar", "Heavy", 1.5)

    # --- Molecule Palette API ---

    def add_molecule(self, template):
        """Insert (or overwrite) a molecule template by name."""
        if isinstance(template, MoleculeTemplate):
            self.molecules[template.name] = template

    def remove_molecule(self, name):
        """Drop a molecule template by name. Silent no-op if absent."""
        self.molecules.pop(name, None)

    def get_molecule(self, name):
        return self.molecules.get(name)

    def get_bond_default(self, mat_a_name, mat_b_name):
        """Return (k, r_eq) defaults for a material pair.

        If the pair has an explicit override in self.bond_defaults that wins.
        Otherwise falls back to a geometry-derived default:
            r_eq = 0.5 * (sigma_a + sigma_b)   (atoms touching at LJ-ish range)
            k    = 200.0                       (stiff enough to hold the bond)

        The fallback resolves through get_material so unknown names still
        produce sensible numbers via the Water/Wall fallback chain.
        """
        key = frozenset({mat_a_name, mat_b_name})
        if key in self.bond_defaults:
            return self.bond_defaults[key]
        mat_a = self.get_material(mat_a_name)
        mat_b = self.get_material(mat_b_name)
        sigma_a = getattr(mat_a, 'sigma', 1.0)
        sigma_b = getattr(mat_b, 'sigma', 1.0)
        r_eq = 0.5 * (sigma_a + sigma_b)
        return (200.0, r_eq)

    def set_bond_default(self, mat_a_name, mat_b_name, k, r_eq):
        """Override the (k, r_eq) defaults for a material pair."""
        key = frozenset({mat_a_name, mat_b_name})
        self.bond_defaults[key] = (float(k), float(r_eq))

    # --- Geometry Queries (New SoC Compliance) ---

    def find_entity_at(self, x, y, radius):
        """
        Finds the index of an entity intersecting the circle (x, y, radius).
        Moved from simulation_geometry.py to keep Model logic encapsulated.
        """
        best_dist = float('inf')
        best_idx = -1

        for i, w in enumerate(self.entities):
            dist = float('inf')

            if isinstance(w, Line):
                if np.array_equal(w.start, w.end):
                    continue
                p1 = w.start
                p2 = w.end
                p3 = np.array([x, y])

                # Point-Line Distance
                d_vec = p2 - p1
                len_sq = np.dot(d_vec, d_vec)
                if len_sq == 0:
                    dist = np.linalg.norm(p3 - p1)
                else:
                    t = np.dot(p3 - p1, d_vec) / len_sq
                    # For infinite ref lines, don't clamp t to segment bounds
                    if not (w.ref and w.infinite):
                        t = max(0, min(1, t))
                    proj = p1 + t * d_vec
                    dist = np.linalg.norm(p3 - proj)
                    
            elif isinstance(w, Circle):
                center_dist = math.hypot(x - w.center[0], y - w.center[1])
                dist = abs(center_dist - w.radius)
                
            elif isinstance(w, Point):
                 dist = math.hypot(x - w.pos[0], y - w.pos[1])

            if dist < radius and dist < best_dist:
                best_dist = dist
                best_idx = i
                
        return best_idx

    # --- Geometry API ---

    def add_line(self, start, end, is_ref=False, anchored=None, material_id="Wall"):
        if anchored is None: anchored = [False, False]
        l = Line(start, end, is_ref, material_id)
        l.anchored = anchored
        self.entities.append(l)
        return len(self.entities) - 1

    def add_circle(self, center, radius, anchored=None, material_id="Wall"):
        if anchored is None: anchored = [False]
        c = Circle(center, radius, material_id)
        c.anchored = anchored
        self.entities.append(c)
        return len(self.entities) - 1

    def update_entity(self, index, **kwargs):
        """
        Generic update for entity properties.
        """
        if 0 <= index < len(self.entities):
            e = self.entities[index]

            # Position updates
            if 'start' in kwargs and hasattr(e, 'start'): e.start[:] = kwargs['start']
            if 'end' in kwargs and hasattr(e, 'end'): e.end[:] = kwargs['end']
            if 'center' in kwargs and hasattr(e, 'center'): e.center[:] = kwargs['center']
            if 'radius' in kwargs and hasattr(e, 'radius'): e.radius = kwargs['radius']

            # Material Assignment
            if 'material_id' in kwargs:
                e.material_id = kwargs['material_id']

            # Physical flag (atomization)
            if 'physical' in kwargs:
                e.physical = kwargs['physical']

            # Anchors
            if 'anchored' in kwargs: e.anchored = kwargs['anchored']

            # Trigger Solve
            self.solve()

    def remove_entity(self, index):
        """
        Removes an entity and any dependent constraints.
        """
        if 0 <= index < len(self.entities):
            self.constraints = [c for c in self.constraints if not self._constraint_involves(c, index)]
            for c in self.constraints:
                self._shift_constraint_indices(c, index)
            self.entities.pop(index)
            self.solve()

    def get_entity(self, index):
        if 0 <= index < len(self.entities):
            return self.entities[index]
        return None

    def toggle_anchor(self, entity_idx, point_idx):
        """
        Toggles the anchor state of a point on an entity.
        Anchored points are fixed in space during constraint solving.
        """
        if not (0 <= entity_idx < len(self.entities)):
            return
        
        entity = self.entities[entity_idx]
        
        if hasattr(entity, 'anchored'):
            if isinstance(entity.anchored, list):
                if 0 <= point_idx < len(entity.anchored):
                    entity.anchored[point_idx] = not entity.anchored[point_idx]
            else:
                # Point entity has single anchored bool
                entity.anchored = not entity.anchored

    # --- Compatibility Alias ---
    
    @property
    def walls(self):
        """Alias for entities - maintains compatibility with CONSTRAINT_DEFS."""
        return self.entities

    # --- Material API ---
    
    def add_material(self, material):
        if isinstance(material, Material):
            self.materials[material.name] = material

    def get_material(self, material_id):
        return self.materials.get(material_id, self.materials.get("Water", self.materials.get("Wall")))

    # --- LJ Cross-Interaction API ---

    def get_lj_epsilon(self, name_a, name_b):
        """Return the effective LJ ε for the pair (name_a, name_b).

        If an override is registered for this pair, return it; otherwise
        fall back to Berthelot's geometric-mean rule on the two materials'
        own ε values. Frozenset key makes the lookup order-independent so
        get_lj_epsilon('A', 'B') == get_lj_epsilon('B', 'A').

        Unknown material names fall back via get_material (which returns
        Water-as-default), so this never raises.
        """
        key = frozenset((name_a, name_b))
        if key in self.lj_cross_overrides:
            return self.lj_cross_overrides[key]
        mat_a = self.get_material(name_a)
        mat_b = self.get_material(name_b)
        return math.sqrt(mat_a.epsilon * mat_b.epsilon)

    def set_lj_cross_override(self, name_a, name_b, epsilon):
        """Register an override for the (name_a, name_b) LJ ε.

        Setting ε_AB < √(ε_AA·ε_BB) makes A and B less attractive than
        Berthelot predicts → demixing / phase separation. Setting
        ε_AB > √(ε_AA·ε_BB) makes them more attractive → mutual solvent.
        """
        self.lj_cross_overrides[frozenset((name_a, name_b))] = float(epsilon)

    def remove_lj_cross_override(self, name_a, name_b):
        """Drop the override for (name_a, name_b). No-op if not set."""
        self.lj_cross_overrides.pop(frozenset((name_a, name_b)), None)

    def build_eps_ij_matrix(self):
        """Return (M, M) numpy float32 matrix of effective LJ ε for every
        pair of currently-registered materials. M is len(self.materials);
        the i-th row/column corresponds to the i-th key in self.materials
        (insertion order, stable across Python dict semantics).

        Atoms record their material's index into this list at spawn time
        (Simulation.atom_material_id) and the LJ kernel indexes the matrix
        directly to get the effective ε for the pair. Per-atom ε_sqrt
        remains the fallback for atoms with material_id < 0 (legacy
        bare-Sim paths, brushes without a material).
        """
        names = list(self.materials.keys())
        n = len(names)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                matrix[i, j] = self.get_lj_epsilon(a, b)
        return matrix

    def get_material_index(self, name):
        """Return the integer index of `name` in self.materials, or -1 if
        the material is not registered. Atoms store this as their stable
        material_id so the LJ kernel can index into build_eps_ij_matrix
        output."""
        try:
            return list(self.materials.keys()).index(name)
        except ValueError:
            return -1

    # --- Constraint API ---

    def add_constraint(self, type_name, indices, value=None):
        data = {'type': type_name, 'indices': indices}
        if value is not None: data['value'] = value
        c = create_constraint(data)
        if c:
            self._handle_constraint_conflicts(c)
            self.constraints.append(c)
            self.solve(iterations=500)
            return c
        return None

    def remove_constraint(self, index):
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
            self.solve()

    def add_constraint_object(self, constraint, solve=True):
        """
        Adds a constraint object, handling conflicts with existing angle constraints.
        For angle-type constraints on the same entities, new replaces old.
        
        Args:
            constraint: The constraint to add
            solve: If True (default), run solver after adding. Set to False
                   for batch operations, then call solve() manually once.
        """
        angle_types = ['PARALLEL', 'PERPENDICULAR', 'HORIZONTAL', 'VERTICAL']
        
        if hasattr(constraint, 'type') and constraint.type in angle_types:
            new_indices = set(constraint.indices) if isinstance(constraint.indices, (list, tuple)) else {constraint.indices}
            
            keep = []
            for c in self.constraints:
                if getattr(c, 'type', '') in angle_types:
                    old_indices = set(c.indices) if isinstance(c.indices, (list, tuple)) else {c.indices}
                    if old_indices == new_indices:
                        continue  # Remove conflicting constraint
                keep.append(c)
            self.constraints = keep
        
        self.constraints.append(constraint)
        
        if solve:
            self.solve(iterations=500)

    def try_create_constraint(self, ctype, entity_idxs, point_idxs):
        """
        Attempts to create a constraint object WITHOUT adding it to the sketch.

        This is used by the command system to create constraints that can be
        properly tracked for undo/redo.

        Args:
            ctype: Constraint type string (e.g., 'LENGTH', 'PARALLEL')
            entity_idxs: List of entity indices
            point_idxs: List of (entity_idx, point_idx) tuples

        Returns:
            The constraint object if valid, None otherwise
        """
        from model.constraint_factory import CONSTRAINT_DEFS

        rules = CONSTRAINT_DEFS.get(ctype, [])

        for rule in rules:
            if len(entity_idxs) == rule['w'] and len(point_idxs) == rule['p']:
                valid = True

                # Type check if required
                if rule.get('t'):
                    for idx in entity_idxs:
                        if idx < len(self.entities) and not isinstance(self.entities[idx], rule['t']):
                            valid = False
                            break

                if valid:
                    # Factory function expects (sketch, walls, pts)
                    return rule['f'](self, entity_idxs, point_idxs)

        return None

    def attempt_apply_constraint(self, ctype, entity_idxs, point_idxs):
        """
        Attempts to create and apply a constraint based on selection.

        WARNING: This method bypasses the command queue and should only be used
        for non-undoable operations. For undoable constraint creation, use
        try_create_constraint() with AddConstraintCommand.

        Args:
            ctype: Constraint type string (e.g., 'LENGTH', 'PARALLEL')
            entity_idxs: List of entity indices
            point_idxs: List of (entity_idx, point_idx) tuples

        Returns:
            True if constraint was successfully applied, False otherwise
        """
        constraint = self.try_create_constraint(ctype, entity_idxs, point_idxs)
        if constraint:
            self.add_constraint_object(constraint)
            return True
        return False

    def set_driver(self, constraint_index, driver_data):
        if 0 <= constraint_index < len(self.constraints):
            c = self.constraints[constraint_index]
            c.driver = driver_data
            if getattr(c, 'value', None) is not None:
                c.base_value = c.value

    def update_drivers(self, time):
        for c in self.constraints:
            if hasattr(c, 'driver') and c.driver:
                d = c.driver
                if c.base_value is None: c.base_value = c.value
                base = c.base_value
                t0 = getattr(c, 'base_time', 0.0)
                if t0 is None: t0 = 0.0
                dt_drive = time - t0
                
                if d['type'] == 'sin':
                    offset = d['amp'] * math.sin(2 * math.pi * d['freq'] * dt_drive + math.radians(d['phase']))
                    c.value = base + offset
                elif d['type'] == 'lin':
                    c.value = base + d['rate'] * dt_drive

    def solve(self, iterations=None):
        if iterations is None:
            iterations = self.solver_iterations
        Solver.solve(self.constraints, self.entities, iterations,
                     use_numba=self.use_numba, interaction_data=self.interaction_data)

    def clear(self):
        self.entities = []
        self.constraints = []
        # Materials persist!

    # --- Private Helpers ---

    def _constraint_involves(self, c, entity_idx):
        for idx in c.indices:
            if isinstance(idx, (list, tuple)):
                if idx[0] == entity_idx: return True
            else:
                if idx == entity_idx: return True
        return False

    def _shift_constraint_indices(self, c, removed_idx):
        new_indices = []
        for idx in c.indices:
            if isinstance(idx, (list, tuple)):
                w_idx, pt_idx = idx
                if w_idx > removed_idx: w_idx -= 1
                new_indices.append((w_idx, pt_idx))
            else:
                val = idx
                if val > removed_idx: val -= 1
                new_indices.append(val)
        c.indices = new_indices

    def _handle_constraint_conflicts(self, new_c):
        angle_types = ['PARALLEL', 'PERPENDICULAR', 'HORIZONTAL', 'VERTICAL']
        if new_c.type in angle_types:
            new_indices = set(new_c.indices) if isinstance(new_c.indices, (list, tuple)) else {new_c.indices}
            keep = []
            for c in self.constraints:
                if c.type in angle_types:
                    old_indices = set(c.indices) if isinstance(c.indices, (list, tuple)) else {c.indices}
                    if old_indices == new_indices: continue 
                keep.append(c)
            self.constraints = keep

    # --- Serialization ---

    def to_dict(self):
        # bond_defaults keys are frozensets of two material names. Serialize
        # as a list of [name_a, name_b, k, r_eq] tuples — JSON-friendly and
        # order-independent on restore (we re-key by frozenset).
        bond_defaults_list = []
        for key, (k, r_eq) in self.bond_defaults.items():
            names = sorted(key)
            # Frozensets of size 1 (self-pair) collapse to one name; pad.
            if len(names) == 1:
                names = [names[0], names[0]]
            bond_defaults_list.append([names[0], names[1], float(k), float(r_eq)])

        # Same trick for lj_cross_overrides — list of [name_a, name_b, eps]
        # triples, JSON-friendly.
        lj_overrides_list = []
        for key, eps in self.lj_cross_overrides.items():
            names = sorted(key)
            if len(names) == 1:
                names = [names[0], names[0]]
            lj_overrides_list.append([names[0], names[1], float(eps)])

        return {
            'entities': [e.to_dict() for e in self.entities],
            'constraints': [c.to_dict() for c in self.constraints],
            'materials': {k: v.to_dict() for k, v in self.materials.items()},
            'molecules': {name: tpl.to_dict() for name, tpl in self.molecules.items()},
            'bond_defaults': bond_defaults_list,
            'lj_cross_overrides': lj_overrides_list,
        }

    def restore(self, data):
        # Local imports
        from model.geometry import Line, Circle, Point
        from model.properties import Material

        if 'materials' in data:
            self.materials = {}
            for k, v in data['materials'].items():
                self.materials[k] = Material.from_dict(v)
        # Ensure essential materials exist (Wall is commonly used for geometry)
        if "Wall" not in self.materials:
            self.materials["Wall"] = Material("Wall", sigma=1.0, epsilon=1.0, color=(100, 100, 120))
        if "Water" not in self.materials:
            self.materials["Water"] = Material("Water", sigma=1.0, epsilon=1.0, color=(50, 150, 255))

        self.entities = []
        for e_data in data.get('entities', []):
            if e_data['type'] == 'line': self.entities.append(Line.from_dict(e_data))
            elif e_data['type'] == 'circle': self.entities.append(Circle.from_dict(e_data))
            elif e_data['type'] == 'point': self.entities.append(Point.from_dict(e_data))

        self.constraints = []
        for c_data in data.get('constraints', []):
            c = create_constraint(c_data)
            if c: self.constraints.append(c)

        # --- Molecules (R2+) ---
        if 'molecules' in data:
            self.molecules = {}
            for name, tpl_data in data['molecules'].items():
                self.molecules[name] = MoleculeTemplate.from_dict(tpl_data)
        else:
            # Pre-R2 save with no molecules — re-seed the starter palette so
            # opening a legacy scene doesn't leave the user without templates.
            self.molecules = {}
            self._seed_default_molecules()

        if 'bond_defaults' in data:
            self.bond_defaults = {}
            for entry in data['bond_defaults']:
                # Tolerate both 4-tuples and dicts
                if isinstance(entry, dict):
                    name_a = entry['mat_a']
                    name_b = entry['mat_b']
                    k = entry['k']
                    r_eq = entry['r_eq']
                else:
                    name_a, name_b, k, r_eq = entry
                self.bond_defaults[frozenset({name_a, name_b})] = (float(k), float(r_eq))
        else:
            self.bond_defaults = {}

        if 'lj_cross_overrides' in data:
            self.lj_cross_overrides = {}
            for entry in data['lj_cross_overrides']:
                if isinstance(entry, dict):
                    name_a = entry['mat_a']
                    name_b = entry['mat_b']
                    eps = entry['epsilon']
                else:
                    name_a, name_b, eps = entry
                self.lj_cross_overrides[frozenset({name_a, name_b})] = float(eps)
        else:
            # Pre-R1 save: no overrides → empty dict, L-B applies everywhere.
            self.lj_cross_overrides = {}