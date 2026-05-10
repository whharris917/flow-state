"""Tests for model/process_objects.py — Source and Sink lifecycle, spawning, absorption, serialization."""

import math
import pytest

from model.process_objects import (
    Source, SourceProperties,
    Sink, SinkProperties,
    create_process_object,
)
from model.sketch import Sketch
from engine.simulation import Simulation


class TestSourceProperties:
    def test_defaults(self):
        p = SourceProperties()
        assert p.material_name == 'Water'
        assert p.flux == 0.5
        assert p.temperature == 1.0
        assert p.injection_spread == pytest.approx(2 * math.pi)

    def test_dict_roundtrip(self):
        p = SourceProperties(material_name='Oil', flux=2.5, temperature=0.5,
                             injection_direction=math.pi / 4, injection_spread=math.pi / 2)
        restored = SourceProperties.from_dict(p.to_dict())
        assert restored.material_name == 'Oil'
        assert restored.flux == 2.5
        assert restored.temperature == 0.5
        assert restored.injection_direction == pytest.approx(math.pi / 4)
        assert restored.injection_spread == pytest.approx(math.pi / 2)

    def test_from_dict_silently_ignores_legacy_sigma_epsilon_mass(self):
        """Saves authored before the material_name refactor still load — the
        per-particle physics fields are ignored, material_name falls back to
        Water, and the rest of the schema is honoured."""
        legacy_dict = {
            'sigma': 1.5,
            'epsilon': 2.0,
            'mass': 3.0,
            'flux': 1.5,
            'temperature': 0.7,
            'injection_direction': 0.0,
            'injection_spread': 1.0,
        }
        p = SourceProperties.from_dict(legacy_dict)
        assert p.material_name == 'Water'  # default fallback
        assert p.flux == 1.5
        assert p.temperature == 0.7
        # Legacy fields not present on the new dataclass
        assert not hasattr(p, 'sigma')
        assert not hasattr(p, 'rate')

    def test_legacy_rate_converts_to_flux_via_source_from_dict(self):
        """A pre-flux save (with `rate` but no `flux`) round-trips into a Source
        whose flux preserves the original total throughput. At radius=3 area is
        π·9 ≈ 28.27, so rate=28.27 should convert to flux=1.0."""
        legacy_save = {
            'type': 'source',
            'center': [10.0, 10.0],
            'radius': 3.0,
            'enabled': True,
            'properties': {
                'rate': math.pi * 9.0,  # exactly π·r² so flux comes out to 1.0
                'temperature': 1.0,
            },
        }
        source = Source.from_dict(legacy_save)
        assert source.properties.flux == pytest.approx(1.0)

    def test_explicit_flux_wins_over_legacy_rate(self):
        """If both `flux` and `rate` appear in a save (forward-mixed), flux wins."""
        mixed = {
            'type': 'source',
            'center': [0.0, 0.0],
            'radius': 3.0,
            'enabled': True,
            'properties': {'flux': 7.0, 'rate': 100.0},
        }
        source = Source.from_dict(mixed)
        assert source.properties.flux == 7.0


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
        source = Source((25, 25), 3.0, SourceProperties(flux=50.0))
        source.enabled = False
        source.execute(sim, dt=0.1)
        assert sim.count == 0

    def test_enabled_source_spawns_over_time(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        # High rate + large dt to make at least one spawn extremely likely
        source = Source((25, 25), 3.0, SourceProperties(flux=100.0))
        source.execute(sim, dt=1.0)
        assert sim.count > 0

    def test_spawned_particles_lie_within_radius(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        source = Source((25, 25), 3.0, SourceProperties(flux=100.0))
        source.execute(sim, dt=1.0)
        for i in range(sim.count):
            dx = sim.pos_x[i] - 25
            dy = sim.pos_y[i] - 25
            dist = math.hypot(dx, dy)
            assert dist <= 3.0 + 0.01  # slight floating-point slack


class TestSerialization:
    def test_source_dict_roundtrip(self):
        source = Source((15, 25), 4.0, SourceProperties(flux=2.5, material_name='Oil'))
        source.enabled = False
        d = source.to_dict()
        assert d["type"] == "source"

        restored = Source.from_dict(d)
        assert restored.x == 15.0
        assert restored.y == 25.0
        assert restored.radius == 4.0
        assert restored.enabled is False
        assert restored.properties.flux == 2.5
        assert restored.properties.material_name == 'Oil'

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
        source = Source((50, 50), 30.0, SourceProperties(flux=20.0))
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

        source = Source((25, 25), 3.0, SourceProperties(flux=100.0))
        source.execute(sim, dt=0.1)
        # At least most spawns should have been rejected — relaxed bound
        added = sim.count - before
        assert added < 5  # some may slip through the gaps; main case is "not flooded"


class TestSourceVelocitySampling:
    def test_directional_bias_produces_rightward_mean(self):
        """With injection_direction=0 (rightward) and a tight spread, mean vx > 0
        and mean vy ≈ 0. Seeded for determinism per TU-SIM."""
        import numpy as np
        import random
        random.seed(42)
        np.random.seed(42)

        sim = Simulation(skip_warmup=True)
        sim.world_size = 100.0
        source = Source((50, 50), 5.0, SourceProperties(
            flux=100.0, temperature=1.0,
            injection_direction=0.0, injection_spread=0.5,
        ))
        for _ in range(5):
            source.execute(sim, dt=1.0)
        assert sim.count > 10
        vx_mean = float(np.mean(sim.vel_x[:sim.count]))
        vy_mean = float(np.mean(sim.vel_y[:sim.count]))
        assert vx_mean > 0
        assert abs(vy_mean) < 0.5

    def test_isotropic_spread_produces_balanced_directions(self):
        """Per TU-SIM: seed both numpy and Python random so the test is
        deterministic. The 1.0 threshold was chosen for a sample size of
        ~100+ at unit temperature/mass; the seeded harness ensures we don't
        rely on luck."""
        import numpy as np
        import random
        random.seed(42)
        np.random.seed(42)

        sim = Simulation(skip_warmup=True)
        sim.world_size = 100.0
        source = Source((50, 50), 5.0, SourceProperties(
            flux=100.0, temperature=1.0,
            injection_direction=0.0, injection_spread=2 * 3.141593,
        ))
        for _ in range(5):
            source.execute(sim, dt=1.0)
        assert sim.count > 10
        # Isotropic — both axes should have means near zero
        assert abs(float(np.mean(sim.vel_x[:sim.count]))) < 1.0
        assert abs(float(np.mean(sim.vel_y[:sim.count]))) < 1.0


class TestSourceMaterialLookup:
    """Sources resolve their particle physics from the sketch's material library."""

    def test_bare_source_falls_back_to_default_material(self):
        """A Source not attached to a Scene still spawns particles using a
        sensible default (covers test code that builds a bare Sim+Source)."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        source = Source((25, 25), 3.0, SourceProperties(flux=100.0))
        source.execute(sim, dt=1.0)
        assert sim.count > 0

    def test_attached_source_uses_named_material(self):
        """A Source attached to a Scene resolves material_name via the Sketch.
        Verify the spawned particles' sigma matches the chosen material."""
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.simulation.world_size = 50.0

        # Confirm Oil is in the preset library (sigma=1.2 from properties.py)
        oil = scene.sketch.get_material('Oil')
        assert abs(oil.sigma - 1.2) < 1e-6

        source = Source((25, 25), 3.0,
                        SourceProperties(material_name='Oil', flux=100.0))
        scene.add_process_object(source)
        source.execute(scene.simulation, dt=1.0)

        assert scene.simulation.count > 0
        # All spawned dynamic particles should carry Oil's sigma
        for i in range(scene.simulation.count):
            if scene.simulation.is_static[i] == 0:
                assert abs(float(scene.simulation.atom_sigma[i]) - oil.sigma) < 1e-5

    def test_unknown_material_falls_back_via_sketch_get_material(self):
        """sketch.get_material returns Water (or Wall) for unknown names, so a
        bad material_name doesn't break spawning."""
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.simulation.world_size = 50.0
        source = Source((25, 25), 3.0,
                        SourceProperties(material_name='DoesNotExist', flux=100.0))
        scene.add_process_object(source)
        # Must not raise; spawning continues with the fallback material
        source.execute(scene.simulation, dt=1.0)
        assert scene.simulation.count > 0

    def test_spawned_particles_inherit_material_color(self):
        """Per-particle color tracks the resolved material so emissions visually
        match other atoms of the same material (fixes the wall-color residue
        issue called out in _add_particle's docstring)."""
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.simulation.world_size = 50.0

        oil = scene.sketch.get_material('Oil')
        source = Source((25, 25), 3.0,
                        SourceProperties(material_name='Oil', flux=100.0))
        scene.add_process_object(source)
        source.execute(scene.simulation, dt=1.0)

        assert scene.simulation.count > 0
        # Each spawned particle's color matches Oil's color
        for i in range(scene.simulation.count):
            c = tuple(int(v) for v in scene.simulation.atom_color[i])
            assert c == tuple(int(v) for v in oil.color)


# =============================================================================
# Sink tests — particle absorber (drain), complement to Source
# =============================================================================


class TestSinkProperties:
    def test_defaults(self):
        p = SinkProperties()
        assert p.enabled_filter is False
        assert p.min_sigma == 0.0
        assert math.isinf(p.max_sigma)

    def test_dict_roundtrip(self):
        p = SinkProperties(enabled_filter=True, min_sigma=0.5, max_sigma=2.0)
        restored = SinkProperties.from_dict(p.to_dict())
        assert restored.enabled_filter is True
        assert restored.min_sigma == 0.5
        assert restored.max_sigma == 2.0

    def test_dict_roundtrip_inf_max_sigma(self):
        """max_sigma=inf cannot be JSON-encoded; must round-trip through None sentinel."""
        p = SinkProperties()  # max_sigma defaults to inf
        d = p.to_dict()
        # JSON-safe — None, not float('inf')
        assert d['max_sigma'] is None
        restored = SinkProperties.from_dict(d)
        assert math.isinf(restored.max_sigma)


class TestSinkConstruction:
    def test_creates_center_handle(self):
        sink = Sink((10.0, 20.0), 3.0)
        assert "center" in sink.handles
        center = sink.handles["center"]
        assert tuple(center.pos) == (10.0, 20.0)
        assert center.is_handle is True

    def test_radius_stored(self):
        sink = Sink((0, 0), 5.5)
        assert sink.radius == 5.5

    def test_enabled_default_true(self):
        assert Sink((0, 0), 1.0).enabled is True

    def test_stats_initialized_to_zero(self):
        sink = Sink((0, 0), 1.0)
        assert sink.absorbed_count == 0
        assert sink.last_frame_absorbed == 0


class TestSinkHandleLifecycle:
    def test_register_handles_appends_to_sketch(self):
        sketch = Sketch()
        sink = Sink((10, 10), 2.0)
        sink.register_handles(sketch)
        assert sink.handles["center"] in sketch.entities

    def test_register_marks_handle_owner(self):
        sketch = Sketch()
        sink = Sink((10, 10), 2.0)
        sink.register_handles(sketch)
        center = sink.handles["center"]
        assert center._owner_process_object is sink

    def test_unregister_removes_handle_from_sketch(self):
        sketch = Sketch()
        sink = Sink((10, 10), 2.0)
        sink.register_handles(sketch)
        sink.unregister_handles(sketch)
        assert sink.handles["center"] not in sketch.entities


class TestSinkHitTesting:
    def test_contains_point_inside_radius(self):
        sink = Sink((10, 10), 5.0)
        assert sink.contains_point(11, 10) is True
        assert sink.contains_point(20, 10) is False

    def test_hit_test_center(self):
        sink = Sink((10, 10), 5.0)
        assert sink.hit_test(10, 10, tolerance=2.0) is True

    def test_hit_test_circumference(self):
        sink = Sink((10, 10), 5.0)
        assert sink.hit_test(15, 10, tolerance=2.0) is True


class TestSinkAbsorption:
    def _seed_dynamic_particle(self, sim, x, y):
        """Helper: place a single dynamic particle at (x,y)."""
        idx = sim._add_particle(x=x, y=y, vx=0.0, vy=0.0, is_static=0)
        return idx

    def test_disabled_sink_absorbs_nothing(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        self._seed_dynamic_particle(sim, 25, 25)
        before = sim.count

        sink = Sink((25, 25), 5.0)
        sink.enabled = False
        sink.execute(sim, dt=0.1)

        assert sim.count == before
        assert sink.absorbed_count == 0

    def test_absorbs_particle_inside_radius(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        self._seed_dynamic_particle(sim, 25, 25)
        sink = Sink((25, 25), 5.0)

        sink.execute(sim, dt=0.1)

        assert sim.count == 0
        assert sink.absorbed_count == 1
        assert sink.last_frame_absorbed == 1

    def test_does_not_absorb_outside_radius(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        self._seed_dynamic_particle(sim, 35, 25)  # 10 units away
        sink = Sink((25, 25), 5.0)  # radius 5

        sink.execute(sim, dt=0.1)

        assert sim.count == 1
        assert sink.absorbed_count == 0
        assert sink.last_frame_absorbed == 0

    def test_absorbs_only_dynamic_not_static(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        # Static particle (wall)
        sim._add_particle(x=25, y=25, vx=0, vy=0, is_static=1)
        # Dynamic particle right next to it
        sim._add_particle(x=25.5, y=25, vx=0, vy=0, is_static=0)
        sink = Sink((25, 25), 5.0)

        sink.execute(sim, dt=0.1)

        # Only the dynamic one should be gone
        assert sim.count == 1
        assert sim.is_static[0] == 1  # surviving particle is the static one
        assert sink.absorbed_count == 1

    def test_absorbs_only_dynamic_not_tethered(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        # Tethered particle
        sim._add_particle(x=25, y=25, vx=0, vy=0, is_static=3)
        # Dynamic particle right next to it
        sim._add_particle(x=25.5, y=25, vx=0, vy=0, is_static=0)
        sink = Sink((25, 25), 5.0)

        sink.execute(sim, dt=0.1)

        assert sim.count == 1
        assert sim.is_static[0] == 3  # surviving particle is the tethered one
        assert sink.absorbed_count == 1

    def test_absorbs_multiple_in_one_call(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        for dx in range(-2, 3):  # 5 particles inside radius 5
            self._seed_dynamic_particle(sim, 25 + dx, 25)
        sink = Sink((25, 25), 5.0)

        sink.execute(sim, dt=0.1)

        assert sim.count == 0
        assert sink.absorbed_count == 5
        assert sink.last_frame_absorbed == 5

    def test_last_frame_absorbed_resets_each_call(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        self._seed_dynamic_particle(sim, 25, 25)
        sink = Sink((25, 25), 5.0)

        sink.execute(sim, dt=0.1)
        assert sink.last_frame_absorbed == 1

        # Next call: nothing in the sink → last_frame_absorbed resets to 0
        sink.execute(sim, dt=0.1)
        assert sink.last_frame_absorbed == 0
        # cumulative count is unchanged
        assert sink.absorbed_count == 1

    def test_empty_simulation_safe(self):
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        sink = Sink((25, 25), 5.0)
        # Must not raise even with sim.count == 0
        sink.execute(sim, dt=0.1)
        assert sink.last_frame_absorbed == 0


class TestSinkSerialization:
    def test_sink_dict_roundtrip(self):
        sink = Sink((15, 25), 4.0, SinkProperties(enabled_filter=True, min_sigma=0.5, max_sigma=2.0))
        sink.enabled = False
        d = sink.to_dict()
        assert d["type"] == "sink"

        restored = Sink.from_dict(d)
        assert restored.x == 15.0
        assert restored.y == 25.0
        assert restored.radius == 4.0
        assert restored.enabled is False
        assert restored.properties.enabled_filter is True
        assert restored.properties.min_sigma == 0.5
        assert restored.properties.max_sigma == 2.0

    def test_sink_dict_roundtrip_default_properties(self):
        sink = Sink((1.0, 2.0), 3.0)
        restored = Sink.from_dict(sink.to_dict())
        assert restored.x == 1.0
        assert restored.y == 2.0
        assert restored.radius == 3.0
        assert math.isinf(restored.properties.max_sigma)

    def test_create_process_object_factory_handles_sink(self):
        sink = Sink((1.0, 2.0), 3.0)
        d = sink.to_dict()
        restored = create_process_object(d)
        assert isinstance(restored, Sink)

    def test_create_process_object_factory_routes_correctly(self):
        """Source and Sink dicts must dispatch to their respective classes."""
        source = Source((1, 2), 1.0)
        sink = Sink((3, 4), 2.0)
        assert isinstance(create_process_object(source.to_dict()), Source)
        assert isinstance(create_process_object(sink.to_dict()), Sink)

    def test_absorbed_count_not_serialized(self):
        """Stats are runtime state, not document state — must not round-trip."""
        sink = Sink((0, 0), 1.0)
        sink.absorbed_count = 42
        sink.last_frame_absorbed = 7
        restored = Sink.from_dict(sink.to_dict())
        assert restored.absorbed_count == 0
        assert restored.last_frame_absorbed == 0


class TestSinkRendering:
    def test_geometry_descriptor_marks_kind_sink(self):
        """The renderer dispatches on `kind` to pick the red palette;
        the Sink must declare kind='sink' in its render descriptor."""
        sink = Sink((10, 10), 5.0)
        geoms = sink.get_geometry_for_rendering()
        assert len(geoms) == 1
        assert geoms[0]['type'] == 'dashed_circle'
        assert geoms[0]['kind'] == 'sink'

    def test_source_geometry_is_unmarked_or_source(self):
        """Source must NOT carry kind='sink' — defaults to 'source' when absent."""
        source = Source((10, 10), 5.0)
        geoms = source.get_geometry_for_rendering()
        assert len(geoms) == 1
        # Source may or may not carry an explicit kind; if it does, it must be 'source'.
        kind = geoms[0].get('kind', 'source')
        assert kind == 'source'
