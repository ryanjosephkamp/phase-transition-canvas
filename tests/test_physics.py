#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Physics Module Tests
================================================================================

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
License:        MIT License
================================================================================
"""

import numpy as np
import pytest
from src.physics import (
    LennardJonesParameters,
    lennard_jones_potential,
    lennard_jones_force_magnitude,
    compute_forces_and_energy,
    calculate_kinetic_energy,
    calculate_temperature,
    apply_periodic_boundaries,
    minimum_image_distance
)


class TestLennardJonesParameters:
    """Tests for LJ parameter dataclass."""
    
    def test_default_parameters(self):
        """Test default LJ parameters."""
        params = LennardJonesParameters()
        assert params.epsilon == 1.0
        assert params.sigma == 1.0
        assert params.cutoff == 2.5
    
    def test_custom_parameters(self):
        """Test custom LJ parameters."""
        params = LennardJonesParameters(epsilon=0.5, sigma=0.8, cutoff=3.0)
        assert params.epsilon == 0.5
        assert params.sigma == 0.8
        assert params.cutoff == 3.0


class TestLennardJonesPotential:
    """Tests for LJ potential function."""
    
    def test_minimum_location(self):
        """Potential minimum should be at r = 2^(1/6) * sigma."""
        epsilon = 1.0
        sigma = 1.0
        r_min = 2**(1/6) * sigma
        
        # Check it's a minimum by comparing to nearby points
        V_min = lennard_jones_potential(r_min, epsilon, sigma)
        V_left = lennard_jones_potential(r_min - 0.01, epsilon, sigma)
        V_right = lennard_jones_potential(r_min + 0.01, epsilon, sigma)
        
        assert V_min < V_left
        assert V_min < V_right
    
    def test_potential_at_sigma(self):
        """V(sigma) should equal 0."""
        epsilon = 1.0
        sigma = 1.0
        V = lennard_jones_potential(sigma, epsilon, sigma)
        assert abs(V) < 1e-10
    
    def test_minimum_value(self):
        """Minimum potential should be -epsilon."""
        epsilon = 1.0
        sigma = 1.0
        r_min = 2**(1/6) * sigma
        V_min = lennard_jones_potential(r_min, epsilon, sigma)
        assert abs(V_min + epsilon) < 1e-10
    
    def test_repulsive_at_short_range(self):
        """Potential should be highly repulsive at short range."""
        epsilon = 1.0
        sigma = 1.0
        r_short = 0.9 * sigma
        V = lennard_jones_potential(r_short, epsilon, sigma)
        assert V > 0  # Positive = repulsive


class TestLennardJonesForce:
    """Tests for LJ force calculation."""
    
    def test_force_zero_at_minimum(self):
        """Force should be zero at potential minimum."""
        epsilon = 1.0
        sigma = 1.0
        r_min = 2**(1/6) * sigma
        F = lennard_jones_force_magnitude(r_min, epsilon, sigma)
        assert abs(F) < 1e-10
    
    def test_repulsive_force_close(self):
        """Force should be positive (repulsive) at short range."""
        epsilon = 1.0
        sigma = 1.0
        r_short = 0.95 * sigma
        F = lennard_jones_force_magnitude(r_short, epsilon, sigma)
        assert F > 0  # Positive = repulsive
    
    def test_attractive_force_far(self):
        """Force should be negative (attractive) beyond minimum."""
        epsilon = 1.0
        sigma = 1.0
        r_far = 1.5 * sigma
        F = lennard_jones_force_magnitude(r_far, epsilon, sigma)
        assert F < 0  # Negative = attractive


class TestForceComputation:
    """Tests for full force computation."""
    
    def test_two_particles_newton_third(self):
        """Forces on two particles should be equal and opposite."""
        positions = np.array([[0.0, 0.0], [1.5, 0.0]])
        box_size = (10.0, 10.0)
        params = LennardJonesParameters()
        
        forces, energy = compute_forces_and_energy(
            positions, box_size, params.epsilon, params.sigma, params.cutoff
        )
        
        # Newton's third law
        assert np.allclose(forces[0], -forces[1], atol=1e-10)
    
    def test_total_force_zero(self):
        """Total force should be approximately zero (momentum conservation)."""
        np.random.seed(42)
        n = 20
        positions = np.random.uniform(0, 10, (n, 2))
        box_size = (10.0, 10.0)
        params = LennardJonesParameters()
        
        forces, energy = compute_forces_and_energy(
            positions, box_size, params.epsilon, params.sigma, params.cutoff
        )
        
        total_force = np.sum(forces, axis=0)
        # Allow small numerical error from parallel computation
        assert np.allclose(total_force, 0, atol=1e-6)
    
    def test_periodic_boundaries(self):
        """Test that periodic boundaries work correctly."""
        # Two particles that are closer through the boundary
        positions = np.array([[0.5, 5.0], [9.5, 5.0]])  # Should be 1.0 apart with PBC
        box_size = (10.0, 10.0)
        params = LennardJonesParameters(cutoff=2.0)
        
        forces, energy = compute_forces_and_energy(
            positions, box_size, params.epsilon, params.sigma, params.cutoff
        )
        
        # Should have non-zero force due to PBC
        assert np.linalg.norm(forces[0]) > 0


class TestKineticEnergy:
    """Tests for kinetic energy calculation."""
    
    def test_stationary_particles(self):
        """Kinetic energy should be zero for stationary particles."""
        velocities = np.zeros((10, 2))
        KE = calculate_kinetic_energy(velocities)
        assert KE == 0.0
    
    def test_kinetic_energy_formula(self):
        """Test KE = 0.5 * sum(v^2) for unit mass."""
        velocities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        KE = calculate_kinetic_energy(velocities)
        expected = 0.5 * (1.0 + 1.0 + 2.0)  # 2.0
        assert abs(KE - expected) < 1e-10


class TestTemperature:
    """Tests for temperature calculation."""
    
    def test_zero_velocity_temperature(self):
        """Temperature should be zero for stationary particles."""
        velocities = np.zeros((10, 2))
        T = calculate_temperature(velocities)
        assert T == 0.0
    
    def test_temperature_positive(self):
        """Temperature should be positive for moving particles."""
        np.random.seed(42)
        velocities = np.random.randn(50, 2)
        T = calculate_temperature(velocities)
        assert T > 0


class TestPeriodicBoundaries:
    """Tests for periodic boundary conditions."""
    
    def test_wrap_positions(self):
        """Test that positions wrap correctly."""
        positions = np.array([[11.0, 5.0], [-1.0, 5.0], [5.0, 12.0]])
        box_size = (10.0, 10.0)
        
        wrapped = apply_periodic_boundaries(positions.copy(), box_size)
        
        # All positions should be in [0, L)
        assert np.all(wrapped >= 0)
        assert np.all(wrapped[:, 0] < box_size[0])
        assert np.all(wrapped[:, 1] < box_size[1])
    
    def test_minimum_image(self):
        """Test minimum image distance calculation."""
        # Points on opposite sides of the box
        pos1 = np.array([1.0, 5.0])
        pos2 = np.array([9.0, 5.0])
        box_size = (10.0, 10.0)
        
        dx, dy, r = minimum_image_distance(pos1, pos2, box_size)
        
        # Should be 2.0 (through the boundary), not 8.0
        assert abs(r - 2.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
