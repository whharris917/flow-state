"""
Property Commands - Entity property modifications

These commands handle entity properties like physical state, dynamic coupling,
material, and animation drivers.
All commands implement the Command ABC from core.commands per C7 compliance.
"""

from core.command_base import Command


class SetPhysicalCommand(Command):
    """Set the physical (atomize) state of an entity."""

    changes_topology = True  # Physical flag affects atomization

    def __init__(self, sketch, entity_index, is_physical,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.is_physical = is_physical
        self.old_physical = None
        self.description = "Set Physical"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_physical = getattr(entity, 'physical', False)
            self.sketch.update_entity(self.entity_index, physical=self.is_physical)
            return True
        return False

    def undo(self):
        if self.old_physical is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                self.sketch.update_entity(self.entity_index, physical=self.old_physical)


class SetEntityDynamicCommand(Command):
    """
    Set the dynamic (two-way coupling) state of an entity.

    When dynamic=True:
    - Entity responds to physics forces from tethered atoms
    - Atoms are tethered (is_static=3) instead of static (is_static=1)
    - Entity has mass, velocity, and can rotate

    When dynamic=False:
    - Entity is immovable (infinite mass)
    - Atoms are static walls
    """

    changes_topology = True  # Dynamic flag changes atom type (static vs tethered)

    def __init__(self, sketch, entity_index, is_dynamic,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.is_dynamic = is_dynamic
        self.old_dynamic = None
        self.description = "Set Dynamic"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_dynamic = getattr(entity, 'dynamic', False)
            entity.dynamic = self.is_dynamic
            return True
        return False

    def undo(self):
        if self.old_dynamic is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                entity = self.sketch.entities[self.entity_index]
                entity.dynamic = self.old_dynamic


class SetMaterialCommand(Command):
    """Set the material of an entity."""

    changes_topology = True  # Material affects atomization (spacing, properties)

    def __init__(self, sketch, entity_index, material_id,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.material_id = material_id
        self.old_material_id = None
        self.description = "Set Material"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_material_id = getattr(entity, 'material_id', 'Wall')
            self.sketch.update_entity(self.entity_index, material_id=self.material_id)
            return True
        return False

    def undo(self):
        if self.old_material_id is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                self.sketch.update_entity(self.entity_index, material_id=self.old_material_id)


class SetDriverCommand(Command):
    """Set animation driver parameters on a constraint."""

    def __init__(self, sketch, constraint_index, driver_data,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.constraint_index = constraint_index
        self.driver_data = driver_data
        self.old_driver_data = None
        self.description = "Set Driver"

    def execute(self) -> bool:
        if 0 <= self.constraint_index < len(self.sketch.constraints):
            constraint = self.sketch.constraints[self.constraint_index]
            self.old_driver_data = getattr(constraint, 'driver', None)
            self.sketch.set_driver(self.constraint_index, self.driver_data)
            return True
        return False

    def undo(self):
        if 0 <= self.constraint_index < len(self.sketch.constraints):
            self.sketch.set_driver(self.constraint_index, self.old_driver_data)
