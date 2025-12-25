"""
ProcessObject Commands - Undo/Redo Support for ProcessObjects

These commands integrate ProcessObjects with the existing Command pattern,
enabling undo/redo for Source creation and deletion.

Handle Deletion Behavior:
- When a handle Point is deleted via normal entity deletion, the owning
  ProcessObject should also be deleted
- This is handled by DeleteEntityCommand checking for is_handle flag
"""

from core.commands import Command
from model.process_objects import Source, create_process_object


class AddSourceCommand(Command):
    """
    Command to add a Source ProcessObject to the scene.
    
    Handles:
    - Creating the Source
    - Registering handles with the Sketch
    - Undo removes Source and unregisters handles
    """
    
    def __init__(self, scene, center, radius, properties=None, 
                 historize=True, supersede=False):
        """
        Create an AddSourceCommand.
        
        Args:
            scene: The Scene instance
            center: (x, y) tuple for center position
            radius: Source radius
            properties: SourceProperties or None for defaults
            historize: If True, add to undo stack
            supersede: If True, replace previous command
        """
        super().__init__(historize, supersede)
        self.scene = scene
        self.center = tuple(center)
        self.radius = float(radius)
        self.properties = properties
        self.source = None
        self.description = "Add Source"
    
    def execute(self) -> bool:
        """Create and add the Source."""
        from model.process_objects import Source, SourceProperties
        
        props = self.properties or SourceProperties()
        self.source = Source(
            center=self.center,
            radius=self.radius,
            properties=props
        )
        
        self.scene.add_process_object(self.source)
        return True
    
    def undo(self):
        """Remove the Source."""
        if self.source is not None:
            self.scene.remove_process_object(self.source)
    
    def redo(self):
        """Re-add the Source."""
        if self.source is not None:
            self.scene.add_process_object(self.source)


class DeleteSourceCommand(Command):
    """
    Command to delete a Source ProcessObject.
    
    Stores the Source data for undo/redo.
    """
    
    def __init__(self, scene, source, historize=True, supersede=False):
        """
        Create a DeleteSourceCommand.
        
        Args:
            scene: The Scene instance
            source: The Source to delete
            historize: If True, add to undo stack
            supersede: If True, replace previous command
        """
        super().__init__(historize, supersede)
        self.scene = scene
        self.source = source
        self.source_data = None  # Saved for undo
        self.description = "Delete Source"
    
    def execute(self) -> bool:
        """Remove the Source, saving its data."""
        if self.source not in self.scene.process_objects:
            return False
        
        # Save state for undo
        self.source_data = self.source.to_dict()
        
        self.scene.remove_process_object(self.source)
        return True
    
    def undo(self):
        """Restore the Source."""
        if self.source_data is not None:
            self.source = create_process_object(self.source_data)
            if self.source:
                self.scene.add_process_object(self.source)
    
    def redo(self):
        """Re-delete the Source."""
        if self.source is not None:
            self.scene.remove_process_object(self.source)


class SetSourceRadiusCommand(Command):
    """
    Command to change a Source's radius.
    
    Supports the supersede pattern for live preview during drag.
    """
    
    def __init__(self, source, new_radius, old_radius=None,
                 historize=True, supersede=False):
        """
        Create a SetSourceRadiusCommand.
        
        Args:
            source: The Source to modify
            new_radius: New radius value
            old_radius: Previous radius (for undo). If None, captures current.
            historize: If True, add to undo stack
            supersede: If True, replace previous command
        """
        super().__init__(historize, supersede)
        self.source = source
        self.new_radius = float(new_radius)
        self.old_radius = float(old_radius) if old_radius is not None else source.radius
        self.description = "Resize Source"
    
    def execute(self) -> bool:
        """Set the new radius."""
        self.source.radius = self.new_radius
        return True
    
    def undo(self):
        """Restore the old radius."""
        self.source.radius = self.old_radius
    
    def merge(self, other) -> bool:
        """Merge sequential radius changes."""
        if isinstance(other, SetSourceRadiusCommand):
            if other.source is self.source and other.historize == self.historize:
                # Keep our old_radius, take their new_radius
                self.new_radius = other.new_radius
                return True
        return False


class SetSourcePropertiesCommand(Command):
    """
    Command to change a Source's properties (rate, temperature, etc.).
    """
    
    def __init__(self, source, new_properties, historize=True, supersede=False):
        """
        Create a SetSourcePropertiesCommand.
        
        Args:
            source: The Source to modify
            new_properties: New SourceProperties instance
            historize: If True, add to undo stack
            supersede: If True, replace previous command
        """
        super().__init__(historize, supersede)
        self.source = source
        self.new_properties = new_properties
        self.old_properties = None  # Captured on execute
        self.description = "Edit Source Properties"
    
    def execute(self) -> bool:
        """Apply new properties."""
        from model.process_objects import SourceProperties
        
        # Save current for undo
        self.old_properties = SourceProperties(
            sigma=self.source.properties.sigma,
            epsilon=self.source.properties.epsilon,
            mass=self.source.properties.mass,
            rate=self.source.properties.rate,
            temperature=self.source.properties.temperature,
            injection_direction=self.source.properties.injection_direction,
            injection_spread=self.source.properties.injection_spread,
        )
        
        self.source.properties = self.new_properties
        return True
    
    def undo(self):
        """Restore old properties."""
        if self.old_properties is not None:
            self.source.properties = self.old_properties


class ToggleSourceEnabledCommand(Command):
    """
    Command to toggle a Source's enabled state.
    """
    
    def __init__(self, source, historize=True, supersede=False):
        """
        Create a ToggleSourceEnabledCommand.
        
        Args:
            source: The Source to toggle
            historize: If True, add to undo stack
            supersede: If True, replace previous command
        """
        super().__init__(historize, supersede)
        self.source = source
        self.description = "Toggle Source"
    
    def execute(self) -> bool:
        """Toggle enabled state."""
        self.source.enabled = not self.source.enabled
        return True
    
    def undo(self):
        """Toggle back."""
        self.source.enabled = not self.source.enabled
