"""Tests for engine/particle_brush.py — paint/erase semantics."""

import numpy as np
import pytest

from engine.particle_brush import ParticleBrush


class TestPaint:
    def test_paint_adds_particles(self, simulation):
        brush = ParticleBrush(simulation)
        added = brush.paint(25.0, 25.0, radius=2.0)
        assert added > 0
        assert simulation.count == added

    def test_paint_outside_world_skips_particles(self, simulation):
        brush = ParticleBrush(simulation)
        # Brush centered just inside world, but radius extends out — atoms
        # outside world bounds are skipped by the bounds check
        simulation.world_size = 10.0
        added = brush.paint(9.5, 9.5, radius=5.0)
        for i in range(simulation.count):
            assert 0 < simulation.pos_x[i] < 10.0
            assert 0 < simulation.pos_y[i] < 10.0

    def test_paint_does_not_overlap_existing_particles(self, simulation):
        brush = ParticleBrush(simulation)
        first = brush.paint(25.0, 25.0, radius=2.0)
        # Painting the same area again should not add many more (overlap rejection)
        second = brush.paint(25.0, 25.0, radius=2.0)
        assert second == 0


class TestErase:
    def test_erase_removes_dynamic_particles(self, simulation):
        brush = ParticleBrush(simulation)
        brush.paint(25.0, 25.0, radius=3.0)
        before = simulation.count
        removed = brush.erase(25.0, 25.0, radius=3.0)
        assert removed > 0
        assert simulation.count < before

    def test_erase_preserves_static_particles(self, simulation):
        # Static atoms (is_static=1) represent walls — brush should not erase them
        simulation._add_particle(25.0, 25.0, is_static=1)
        brush = ParticleBrush(simulation)
        brush.erase(25.0, 25.0, radius=5.0)
        assert simulation.count == 1
        assert simulation.is_static[0] == 1

    def test_erase_preserves_tethered_particles(self, simulation):
        simulation._add_particle(25.0, 25.0, is_static=3)
        brush = ParticleBrush(simulation)
        brush.erase(25.0, 25.0, radius=5.0)
        assert simulation.count == 1
        assert simulation.is_static[0] == 3


class TestSpray:
    def test_spray_adds_particles(self, simulation):
        brush = ParticleBrush(simulation)
        added = brush.spray(25.0, 25.0, radius=2.0, density=1.0)
        # density=1 over pi*r^2 = ~12.5 attempts; some succeed
        assert added > 0
        assert simulation.count == added

    def test_spray_respects_world_bounds(self, simulation):
        simulation.world_size = 10.0
        brush = ParticleBrush(simulation)
        # Spray at edge — radius extends out
        brush.spray(9.5, 9.5, radius=5.0, density=1.0)
        for i in range(simulation.count):
            assert 0 < simulation.pos_x[i] < 10.0
            assert 0 < simulation.pos_y[i] < 10.0


class TestFillRect:
    def test_fill_rect_fills_region(self, simulation):
        brush = ParticleBrush(simulation)
        added = brush.fill_rect(20.0, 20.0, 30.0, 30.0)
        assert added > 0
        # All particles should lie within the rectangle
        for i in range(simulation.count):
            assert 20.0 <= simulation.pos_x[i] <= 30.0
            assert 20.0 <= simulation.pos_y[i] <= 30.0

    def test_fill_rect_clamps_to_world_bounds(self, simulation):
        simulation.world_size = 10.0
        brush = ParticleBrush(simulation)
        brush.fill_rect(-5, -5, 100, 100)  # extends way past world
        for i in range(simulation.count):
            assert 0 <= simulation.pos_x[i] <= 10.0
            assert 0 <= simulation.pos_y[i] <= 10.0
