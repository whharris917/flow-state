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