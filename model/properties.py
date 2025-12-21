class Material:
    """
    Defines the physical and visual properties of a geometric entity.
    This separates 'Physics' from 'Geometry'.
    """
    def __init__(self, name, sigma=1.0, epsilon=1.0, spacing=None, color=(200, 200, 200), physical=True):
        self.name = name
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        # Auto-calculate spacing if not provided (0.7 * sigma is standard for fluid walls)
        self.spacing = float(spacing) if spacing is not None else (0.7 * self.sigma)
        self.color = tuple(color)
        self.physical = physical  # True = Collidable, False = Ghost/Guide

    def to_dict(self):
        return {
            'name': self.name,
            'sigma': self.sigma,
            'epsilon': self.epsilon,
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
            spacing=data.get('spacing'),
            color=tuple(data.get('color', (200, 200, 200))),
            physical=data.get('physical', True)
        )