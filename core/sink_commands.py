"""
Sink Commands - Undo/Redo Support for Sink ProcessObjects

Mirrors core/source_commands.py for the Sink (particle absorber) peer.
"""

from core.commands import Command
from model.process_objects import Sink, SinkProperties, create_process_object


class AddSinkCommand(Command):
    """
    Command to add a Sink ProcessObject to the scene.
    """

    def __init__(self, scene, center, radius, properties=None,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.scene = scene
        self.center = tuple(center)
        self.radius = float(radius)
        self.properties = properties
        self.sink = None
        self.description = "Add Sink"

    def execute(self) -> bool:
        """Create and add the Sink."""
        props = self.properties or SinkProperties()
        self.sink = Sink(
            center=self.center,
            radius=self.radius,
            properties=props,
        )

        self.scene.add_process_object(self.sink)
        return True

    def undo(self):
        """Remove the Sink."""
        if self.sink is not None:
            self.scene.remove_process_object(self.sink)

    def redo(self):
        """Re-add the Sink."""
        if self.sink is not None:
            self.scene.add_process_object(self.sink)


class DeleteSinkCommand(Command):
    """
    Command to delete a Sink ProcessObject.
    """

    def __init__(self, scene, sink, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.scene = scene
        self.sink = sink
        self.sink_data = None
        self.description = "Delete Sink"

    def execute(self) -> bool:
        """Remove the Sink, saving its data."""
        if self.sink not in self.scene.process_objects:
            return False

        self.sink_data = self.sink.to_dict()
        self.scene.remove_process_object(self.sink)
        return True

    def undo(self):
        """Restore the Sink."""
        if self.sink_data is not None:
            self.sink = create_process_object(self.sink_data)
            if self.sink:
                self.scene.add_process_object(self.sink)

    def redo(self):
        """Re-delete the Sink."""
        if self.sink is not None:
            self.scene.remove_process_object(self.sink)


class SetSinkRadiusCommand(Command):
    """
    Command to change a Sink's radius. Supports supersede for live drag.
    """

    def __init__(self, sink, new_radius, old_radius=None,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sink = sink
        self.new_radius = float(new_radius)
        self.old_radius = float(old_radius) if old_radius is not None else sink.radius
        self.description = "Resize Sink"

    def execute(self) -> bool:
        self.sink.radius = self.new_radius
        return True

    def undo(self):
        self.sink.radius = self.old_radius

    def merge(self, other) -> bool:
        if isinstance(other, SetSinkRadiusCommand):
            if other.sink is self.sink and other.historize == self.historize:
                self.new_radius = other.new_radius
                return True
        return False


class ToggleSinkEnabledCommand(Command):
    """
    Command to toggle a Sink's enabled state.
    """

    def __init__(self, sink, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sink = sink
        self.description = "Toggle Sink"

    def execute(self) -> bool:
        self.sink.enabled = not self.sink.enabled
        return True

    def undo(self):
        self.sink.enabled = not self.sink.enabled
