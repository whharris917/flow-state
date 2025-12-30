import core.config as config


class Material:
    """
    Defines the physical and visual properties of a geometric entity or fluid.
    This separates 'Physics' from 'Geometry'.

    Properties:
        name: Display name for the material
        sigma: Particle size (LJ sigma parameter)
        epsilon: Interaction strength (LJ epsilon parameter)
        mass: Particle mass (affects inertia and momentum)
        spacing: Distance between atoms when atomizing geometry
        color: RGB tuple for rendering
        physical: If True, collidable; if False, guide/reference only
    """
    def __init__(self, name, sigma=1.0, epsilon=1.0, mass=None, spacing=None,
                 color=(200, 200, 200), physical=True):
        self.name = name
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.mass = float(mass) if mass is not None else config.ATOM_MASS
        # Auto-calculate spacing if not provided (0.7 * sigma is standard for fluid walls)
        self.spacing = float(spacing) if spacing is not None else (0.7 * self.sigma)
        self.color = tuple(color)
        self.physical = physical  # True = Collidable, False = Guide (non-atomized)

    def copy(self):
        """Create a copy of this material."""
        return Material(
            name=self.name,
            sigma=self.sigma,
            epsilon=self.epsilon,
            mass=self.mass,
            spacing=self.spacing,
            color=self.color,
            physical=self.physical
        )

    def to_dict(self):
        return {
            'name': self.name,
            'sigma': self.sigma,
            'epsilon': self.epsilon,
            'mass': self.mass,
            'spacing': self.spacing,
            'color': self.color,
            'physical': self.physical
        }

    @staticmethod
    def from_dict(data):
        return Material(
            name=data['name'],
            sigma=data.get('sigma', 1.0),
            epsilon=data.get('epsilon', 1.0),
            mass=data.get('mass'),
            spacing=data.get('spacing'),
            color=tuple(data.get('color', (200, 200, 200))),
            physical=data.get('physical', True)
        )


# Preset material library
PRESET_MATERIALS = {
    'Water': Material('Water', sigma=1.0, epsilon=1.0, mass=1.0, color=(50, 150, 255)),
    'Oil': Material('Oil', sigma=1.2, epsilon=0.8, mass=0.9, color=(180, 140, 60)),
    'Mercury': Material('Mercury', sigma=0.8, epsilon=2.0, mass=13.5, color=(180, 180, 190)),
    'Honey': Material('Honey', sigma=1.5, epsilon=1.5, mass=1.4, color=(255, 180, 50)),
    'Wall': Material('Wall', sigma=1.0, epsilon=1.0, mass=1.0, color=(100, 100, 120)),
}


class MaterialManager:
    """
    Single source of truth for materials in the application.

    Responsibilities:
    - Provides access to the master material library (sketch.materials)
    - Tracks original values to detect unsaved changes
    - Provides "(modified)" indicator when current values differ from saved
    - Handles save/revert operations
    - Triggers rebuild callbacks when reverting

    Design:
    - Live edits modify sketch.materials directly (for immediate visual feedback)
    - Original snapshots are stored when editing begins
    - has_pending_changes() compares current vs original
    - revert() restores from original snapshot
    - save() clears the original snapshot (current becomes new "saved" state)
    """

    def __init__(self, sketch, on_rebuild=None):
        """
        Initialize the MaterialManager.

        Args:
            sketch: The Sketch instance containing the master materials dict
            on_rebuild: Callback function to trigger physics rebuild
        """
        self.sketch = sketch
        self.on_rebuild = on_rebuild
        # Original snapshots: material_id -> Material snapshot before editing
        self._originals = {}

    @property
    def materials(self):
        """Access the master material library."""
        return self.sketch.materials

    def get_material_names(self):
        """Get list of all material names."""
        return list(self.sketch.materials.keys())

    def get_material(self, material_id):
        """
        Get a material by ID from the library.

        Args:
            material_id: Name of the material

        Returns:
            Material object, or None if not found
        """
        return self.sketch.materials.get(material_id)

    def begin_editing(self, material_id):
        """
        Begin editing a material - stores original values for comparison/revert.

        Call this when user selects an entity to edit its material.

        Args:
            material_id: Name of the material to edit
        """
        if material_id and material_id not in self._originals:
            original = self.sketch.materials.get(material_id)
            if original:
                self._originals[material_id] = original.copy()

    def has_pending_changes(self, material_id):
        """
        Check if a material has unsaved changes (differs from original).

        Args:
            material_id: Name of the material

        Returns:
            True if current values differ from original snapshot
        """
        if material_id not in self._originals:
            return False

        original = self._originals[material_id]
        current = self.sketch.materials.get(material_id)

        if not current:
            return False

        # Compare key properties
        return (
            abs(current.sigma - original.sigma) > 0.001 or
            abs(current.epsilon - original.epsilon) > 0.001 or
            abs(current.mass - original.mass) > 0.001 or
            current.color != original.color
        )

    def get_display_name(self, material_id):
        """
        Get display name for a material with "(modified)" suffix if changed.

        Args:
            material_id: Name of the material

        Returns:
            Display name string
        """
        if self.has_pending_changes(material_id):
            return f"{material_id} (modified)"
        return material_id

    def save(self, material_id):
        """
        Save changes - clears original snapshot so current values become "saved".

        Args:
            material_id: Name of the material to save

        Returns:
            True if there were changes to save, False otherwise
        """
        if material_id not in self._originals:
            return False

        # Current values in sketch.materials are already the "live" values
        # Just clear the original snapshot to mark as saved
        del self._originals[material_id]
        return True

    def revert(self, material_id):
        """
        Revert to original values - restores from snapshot.

        Args:
            material_id: Name of the material to revert

        Returns:
            True if reverted, False if no original to revert to
        """
        if material_id not in self._originals:
            return False

        original = self._originals[material_id]
        current = self.sketch.materials.get(material_id)

        if current:
            # Restore original values
            current.sigma = original.sigma
            current.epsilon = original.epsilon
            current.mass = original.mass
            current.spacing = original.spacing
            current.color = original.color
            current.physical = original.physical

        del self._originals[material_id]

        # Trigger rebuild to show reverted values
        if self.on_rebuild:
            self.on_rebuild()

        return True

    def revert_all(self):
        """Revert all pending changes."""
        material_ids = list(self._originals.keys())
        for mat_id in material_ids:
            self.revert(mat_id)

    def save_as_new(self, old_material_id, new_name):
        """
        Save current values as a new material.

        Args:
            old_material_id: ID of the material being edited
            new_name: Name for the new material

        Returns:
            new_name if saved successfully, None if failed
        """
        if not new_name or new_name in self.sketch.materials:
            return None  # Invalid or duplicate name

        current = self.sketch.materials.get(old_material_id)
        if not current:
            return None

        new_mat = current.copy()
        new_mat.name = new_name
        self.sketch.materials[new_name] = new_mat

        # Clear original snapshot for old material (values saved to new)
        if old_material_id in self._originals:
            # Revert the old material to its original state
            self.revert(old_material_id)

        return new_name

    def name_exists(self, name):
        """Check if a material name already exists."""
        return name in self.sketch.materials