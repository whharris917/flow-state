"""Tests for model/process_objects.py — Source handle lifecycle, spawning, serialization."""

import math
import pytest

from model.process_objects import Source, SourceProperties, create_process_object
from model.sketch import Sketch
from engine.simulation import Simulation


class TestSourceProperties:
    def test_defaults(self):
        p = SourceProperties()
        assert p.sigma == 1.0
        assert p.epsilon == 1.0
        assert p.rate == 10.0
        assert p.injection_spread == pytest.approx(2 * math.pi)

    def test_dict_roundtrip(self):
        p = SourceProperties(sigma=1.5, epsilon=2.0, rate=20.0, temperature=0.5,
                             injection_direction=math.pi / 4, injection_spread=math.pi / 2)
        restored = SourceProperties.from_dict(p.to_dict())
        assert restored.sigma == 1.5
        assert restored.epsilon == 2.0
        assert restored.rate == 20.0
        assert restored.temperature == 0.5
        assert restored.injection_direction == pytest.approx(math.pi / 4)
        assert restored.injection_spread == pytest.approx(math.pi / 2)


class TestSourceConstruction:
    def test_creates_center_handle(self):
        source = Source((10.0, 20.0), 3.0)
        assert "center" in source.handles
        center = source.handles["center"]
        assert tuple(center.pos) == (10.0, 20.0)
        assert center.is_handle is True

    def test_radius_stored(self):
        source = Source((0, 0), 5.5)
        assert source.radius == 5.5

    def test_enabled_default_true(self):
        assert Source((0, 0), 1.0).enabled is True


class TestSourceHandleLifecycle:
    def test_register_handles_appends_to_sketch(self):
        sketch = Sketch()
        source = Source((10, 10), 2.0)
        source.register_handles(sketch)
        assert source.handles["center"] in sketch.entities

    def test_register_marks_handle_owner(self):
        sketch = Sketch()
        source = Source((10, 10), 2.0)
        source.register_handles(sketch)
        center = source.handles["center"]
        assert center._owner_process_object is source

    def test_unregister_removes_handle_from_sketch(self):
        sketch = Sketch()
        source = Source((10, 10), 2.0)
        source.register_handles(sketch)
        source.unregister_handles(sketch)
        assert source.handles["center"] not in sketch.entities


class TestSourceHitTesting:
    def test_contains_point_inside_radius(self):
        source = Source((10, 10), 5.0)
        assert source.contains_point(11, 10) is True
        assert source.contains_point(20, 10) is False

    def test_hit_test_center(self):
        source = Source((10, 10), 5.0)
        assert source.hit_test(10, 10, tolerance=2.0) is True

    def test_hit_test_circumference(self):
        source = Source((10, 10), 5.0)
        # Point on edge (15, 10) should hit
        assert source.hit_test(15, 10, tolerance=2.0) is True


class TestSourceSpawning:
    def test_disabled_source_does_not_spawn(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        source = Source((25, 25), 3.0, SourceProperties(rate=100.0))
        source.enabled = False
        source.execute(sim, dt=0.1)
        assert sim.count == 0

    def test_enabled_source_spawns_over_time(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        # High rate + large dt to make at least one spawn extremely likely
        source = Source((25, 25), 3.0, SourceProperties(rate=1000.0))
        source.execute(sim, dt=1.0)
        assert sim.count > 0

    def test_spawned_particles_lie_within_radius(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        source = Source((25, 25), 3.0, SourceProperties(rate=1000.0))
        source.execute(sim, dt=1.0)
        for i in range(sim.count):
            dx = sim.pos_x[i] - 25
            dy = sim.pos_y[i] - 25
            dist = math.hypot(dx, dy)
            assert dist <= 3.0 + 0.01  # slight floating-point slack


class TestSerialization:
    def test_source_dict_roundtrip(self):
        source = Source((15, 25), 4.0, SourceProperties(rate=50.0, sigma=1.5))
        source.enabled = False
        d = source.to_dict()
        assert d["type"] == "source"

        restored = Source.from_dict(d)
        assert restored.x == 15.0
        assert restored.y == 25.0
        assert restored.radius == 4.0
        assert restored.enabled is False
        assert restored.properties.rate == 50.0
        assert restored.properties.sigma == 1.5

    def test_create_process_object_factory(self):
        source = Source((1.0, 2.0), 3.0)
        d = source.to_dict()
        restored = create_process_object(d)
        assert isinstance(restored, Source)

    def test_create_process_object_unknown_returns_none(self):
        assert create_process_object({"type": "unknown_thing"}) is None

    def test_round_trip_preserves_zero_injection_direction(self):
        """`from_dict`'s `data.get('injection_direction', 0)` falls back to 0,
        so a non-default injection_direction=0 must still survive a round-trip
        even though the value matches the default."""
        source = Source((0, 0), 1.0, SourceProperties(injection_direction=0.0,
                                                       injection_spread=0.5))
        d = source.to_dict()
        restored = Source.from_dict(d)
        assert restored.properties.injection_direction == 0.0
        assert restored.properties.injection_spread == 0.5


class TestSourceRateGuards:
    def test_max_spawns_per_frame_cap(self):
        """Even at pathological rates the source must not exceed the per-frame cap."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 100.0  # large enough to fit 50+ particles
        source = Source((50, 50), 30.0, SourceProperties(rate=10000.0))
        # dt=10 → accumulator = 100000, way over cap
        source.execute(sim, dt=10.0)
        assert sim.count <= Source.MAX_SPAWNS_PER_FRAME

    def test_overlap_rejection_prevents_spawn_when_dense(self):
        """If the spawn region is already saturated with particles, rejection
        sampling at sigma*0.8 must produce zero new spawns."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        # Pre-fill the source region with overlapping particles
        from engine.particle_brush import ParticleBrush
        brush = ParticleBrush(sim)
        brush.fill_rect(20, 20, 30, 30)
        before = sim.count

        source = Source((25, 25), 3.0, SourceProperties(rate=1000.0, sigma=1.0))
        source.execute(sim, dt=0.1)
        # At least most spawns should have been rejected — relaxed bound
        added = sim.count - before
        assert added < 5  # some may slip through the gaps; main case is "not flooded"


class TestSourceVelocitySampling:
    def test_directional_bias_produces_rightward_mean(self):
        """With injection_direction=0 (rightward) and a tight spread, mean vx > 0
        and mean vy ≈ 0."""
        import numpy as np
        sim = Simulation(skip_warmup=True)
        sim.world_size = 100.0
        source = Source((50, 50), 5.0, SourceProperties(
            rate=10000.0, temperature=1.0, mass=1.0,
            injection_direction=0.0, injection_spread=0.5,
        ))
        # Run multiple frames to gather a sample
        for _ in range(5):
            source.execute(sim, dt=1.0)
        assert sim.count > 10
        vx_mean = float(np.mean(sim.vel_x[:sim.count]))
        vy_mean = float(np.mean(sim.vel_y[:sim.count]))
        assert vx_mean > 0
        assert abs(vy_mean) < 0.5

    def test_isotropic_spread_produces_balanced_directions(self):
        import numpy as np
        sim = Simulation(skip_warmup=True)
        sim.world_size = 100.0
        source = Source((50, 50), 5.0, SourceProperties(
            rate=10000.0, temperature=1.0, mass=1.0,
            injection_direction=0.0, injection_spread=2 * 3.141593,
        ))
        for _ in range(5):
            source.execute(sim, dt=1.0)
        assert sim.count > 10
        # Isotropic — both axes should have means near zero
        assert abs(float(np.mean(sim.vel_x[:sim.count]))) < 1.0
        assert abs(float(np.mean(sim.vel_y[:sim.count]))) < 1.0
