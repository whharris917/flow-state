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
    - Manages per-entity preview overrides for selection-aware editing

    Design:
    - When NO entity is selected: Live edits modify sketch.materials directly
    - When entity IS selected: Edits go to per-entity preview overrides
    - Compiler uses get_effective_material() to get the right values per entity
    - Original snapshots track baseline for revert/modified detection
    - Preview overrides are cleared on save/revert/deselect

    Preview Modes:
    - Global preview (no selection): Modifies library, affects all entities with that material
    - Local preview (entity selected): Creates override, affects only selected entity
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
        # Per-entity preview overrides: entity_idx -> Material with preview values
        self._entity_overrides = {}
        # Track which entities are currently being previewed
        self._previewing_entities = set()

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

    # =========================================================================
    # Per-Entity Preview Override System
    # =========================================================================

    def begin_entity_preview(self, entity_idx, material_id):
        """
        Begin previewing material changes for a specific entity.

        Creates a preview override with a copy of the current material values.
        Changes to this override will only affect this entity during preview.

        Args:
            entity_idx: Index of the entity in sketch.entities
            material_id: The material_id of the entity
        """
        if entity_idx in self._entity_overrides:
            return  # Already previewing this entity

        # Get the library material to copy
        mat = self.sketch.materials.get(material_id)
        if not mat:
            return

        # Create override with current library values
        self._entity_overrides[entity_idx] = mat.copy()
        self._previewing_entities.add(entity_idx)

        # Also store original for modified detection (if not already stored)
        if material_id not in self._originals:
            self._originals[material_id] = mat.copy()

    def update_entity_preview(self, entity_idx, **kwargs):
        """
        Update preview values for a specific entity.

        Args:
            entity_idx: Index of the entity being previewed
            **kwargs: Material properties to update (sigma, epsilon, mass, color, etc.)
        """
        override = self._entity_overrides.get(entity_idx)
        if not override:
            return

        for key, value in kwargs.items():
            if hasattr(override, key):
                setattr(override, key, value)

        # Auto-update spacing based on sigma
        if 'sigma' in kwargs:
            override.spacing = 0.7 * override.sigma

    def get_entity_override(self, entity_idx):
        """
        Get the preview override for an entity, if any.

        Args:
            entity_idx: Index of the entity

        Returns:
            Material override or None if not previewing
        """
        return self._entity_overrides.get(entity_idx)

    def get_effective_material(self, entity_idx, material_id):
        """
        Get the effective material for an entity, considering preview overrides.

        This is the main method the Compiler should use to get material properties.
        Returns the preview override if the entity is being previewed, otherwise
        returns the library material.

        Args:
            entity_idx: Index of the entity in sketch.entities
            material_id: The entity's material_id

        Returns:
            Material object (either override or library material)
        """
        # Check for per-entity override first
        override = self._entity_overrides.get(entity_idx)
        if override:
            return override

        # Fall back to library material
        return self.sketch.materials.get(material_id)

    def has_entity_override(self, entity_idx):
        """Check if an entity has a preview override."""
        return entity_idx in self._entity_overrides

    def is_entity_modified(self, entity_idx, material_id):
        """
        Check if an entity's preview values differ from the library material.

        Args:
            entity_idx: Index of the entity
            material_id: The entity's material_id

        Returns:
            True if override exists and differs from library
        """
        override = self._entity_overrides.get(entity_idx)
        if not override:
            return False

        library_mat = self.sketch.materials.get(material_id)
        if not library_mat:
            return False

        # Compare key properties
        return (
            abs(override.sigma - library_mat.sigma) > 0.001 or
            abs(override.epsilon - library_mat.epsilon) > 0.001 or
            abs(override.mass - library_mat.mass) > 0.001 or
            override.color != library_mat.color
        )

    def clear_entity_preview(self, entity_idx):
        """
        Clear preview override for an entity (discard changes).

        Args:
            entity_idx: Index of the entity
        """
        if entity_idx in self._entity_overrides:
            del self._entity_overrides[entity_idx]
        self._previewing_entities.discard(entity_idx)

    def clear_all_entity_previews(self):
        """Clear all per-entity preview overrides."""
        self._entity_overrides.clear()
        self._previewing_entities.clear()

    def commit_entity_preview(self, entity_idx, material_id):
        """
        Commit entity preview to the library material (Save).

        Copies the preview values to the library material and clears the override.
        This affects all entities using that material.

        Args:
            entity_idx: Index of the entity
            material_id: The material_id to update in the library

        Returns:
            True if committed, False if no override exists
        """
        override = self._entity_overrides.get(entity_idx)
        if not override:
            return False

        library_mat = self.sketch.materials.get(material_id)
        if not library_mat:
            return False

        # Copy preview values to library material
        library_mat.sigma = override.sigma
        library_mat.epsilon = override.epsilon
        library_mat.mass = override.mass
        library_mat.spacing = override.spacing
        library_mat.color = override.color

        # Clear the override and original snapshot
        self.clear_entity_preview(entity_idx)
        if material_id in self._originals:
            del self._originals[material_id]

        return True

    def save_entity_preview_as_new(self, entity_idx, old_material_id, new_name):
        """
        Save entity preview as a new material and assign it to the entity.

        Creates a new material from the preview values, adds it to the library,
        and clears the preview override. The entity's material_id should be
        updated by the caller.

        Args:
            entity_idx: Index of the entity
            old_material_id: The original material_id (for reverting)
            new_name: Name for the new material

        Returns:
            The new material name if successful, None if failed
        """
        if not new_name or new_name in self.sketch.materials:
            return None  # Invalid or duplicate name

        override = self._entity_overrides.get(entity_idx)
        if not override:
            return None

        # Create new material from override values
        new_mat = override.copy()
        new_mat.name = new_name
        self.sketch.materials[new_name] = new_mat

        # Clear the entity preview (values are now in new material)
        self.clear_entity_preview(entity_idx)

        # Clear the original snapshot for old material (it wasn't actually modified)
        if old_material_id in self._originals:
            del self._originals[old_material_id]

        return new_name