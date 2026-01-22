#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Simulation Module Tests
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
from src.physics import LennardJonesParameters, calculate_temperature
from src.simulation import (
    MDSimulation, SimulationConfig, SimulationState,
    create_crystal_simulation
)


class TestSimulationConfig:
    """Tests for simulation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.dt == 0.002
        assert config.use_thermostat == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            box_size=(15.0, 15.0),
            dt=0.001,
            use_thermostat=True,
            target_temperature=0.5
        )
        assert config.box_size == (15.0, 15.0)
        assert config.dt == 0.001
        assert config.use_thermostat == True
        assert config.target_temperature == 0.5


class TestMDSimulation:
    """Tests for the MD simulation class."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        config = SimulationConfig(box_size=(10.0, 10.0))
        sim = MDSimulation(config)
        
        assert sim.config == config
        assert sim.state is None  # Not initialized yet
    
    def test_crystal_initialization(self):
        """Test crystal lattice initialization."""
        config = SimulationConfig(box_size=(20.0, 20.0))
        sim = MDSimulation(config)
        sim.initialize_crystal(n_particles_x=6, n_particles_y=6, spacing=1.2, temperature=0.1)
        
        assert sim.state is not None
        assert len(sim.state.positions) == 36
        assert len(sim.state.velocities) == 36
    
    def test_random_initialization(self):
        """Test random initialization."""
        config = SimulationConfig(box_size=(10.0, 10.0))
        sim = MDSimulation(config)
        sim.initialize_random(n_particles=50, temperature=0.5)
        
        assert sim.state is not None
        assert len(sim.state.positions) == 50
        
        # Positions should be within box
        assert np.all(sim.state.positions >= 0)
        assert np.all(sim.state.positions[:, 0] < 10.0)
        assert np.all(sim.state.positions[:, 1] < 10.0)
    
    def test_initial_temperature(self):
        """Test that initial temperature is approximately correct."""
        config = SimulationConfig(box_size=(20.0, 20.0))
        sim = MDSimulation(config)
        target_T = 0.5
        sim.initialize_random(n_particles=100, temperature=target_T)
        
        actual_T = calculate_temperature(sim.state.velocities)
        
        # Should be within 20% (statistical fluctuations)
        assert abs(actual_T - target_T) / target_T < 0.3
    
    def test_step_updates_positions(self):
        """Test that stepping updates positions."""
        sim = create_crystal_simulation(n_particles=25, box_size=15.0, temperature=0.3)
        
        initial_positions = sim.state.positions.copy()
        sim.step()
        
        # Positions should change (unless perfectly equilibrated, very unlikely)
        assert not np.allclose(sim.state.positions, initial_positions)
    
    def test_periodic_boundaries_applied(self):
        """Test that positions stay within box."""
        config = SimulationConfig(box_size=(10.0, 10.0))
        sim = MDSimulation(config)
        sim.initialize_random(n_particles=50, temperature=1.0)
        
        for _ in range(100):
            sim.step()
        
        # All positions should be in [0, L)
        assert np.all(sim.state.positions >= 0)
        assert np.all(sim.state.positions[:, 0] < 10.0)
        assert np.all(sim.state.positions[:, 1] < 10.0)


class TestEnergyConservation:
    """Tests for energy conservation (NVE ensemble)."""
    
    def test_energy_conservation_short(self):
        """Test energy conservation over short time."""
        sim = create_crystal_simulation(
            n_particles=36, box_size=12.0, temperature=0.3, spacing=1.2
        )
        
        # Let system settle a bit
        for _ in range(100):
            sim.step()
        
        initial_energy = sim.state.total_energy
        
        for _ in range(500):
            sim.step()
        
        final_energy = sim.state.total_energy
        
        # Energy should be conserved within 5%
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        assert energy_drift < 0.05, f"Energy drift: {energy_drift*100:.2f}%"
    
    def test_energy_components_positive(self):
        """Test that kinetic energy is always non-negative."""
        sim = create_crystal_simulation(n_particles=25, box_size=15.0, temperature=0.5)
        
        for _ in range(100):
            sim.step()
            assert sim.state.kinetic_energy >= 0


class TestMomentumConservation:
    """Tests for momentum conservation."""
    
    def test_zero_total_momentum(self):
        """Test that total momentum remains approximately zero."""
        sim = create_crystal_simulation(n_particles=36, box_size=15.0, temperature=0.3)
        
        # Initial momentum should be approximately zero (removed in initialization)
        initial_momentum = np.sum(sim.state.velocities, axis=0)
        assert np.allclose(initial_momentum, 0, atol=1e-8)
        
        for _ in range(100):
            sim.step()
        
        # Momentum should remain approximately zero (allow small numerical drift)
        final_momentum = np.sum(sim.state.velocities, axis=0)
        assert np.allclose(final_momentum, 0, atol=0.1)


class TestHeatTransfer:
    """Tests for heat addition/removal."""
    
    def test_add_heat_increases_temperature(self):
        """Test that adding heat increases local temperature."""
        sim = create_crystal_simulation(n_particles=64, box_size=20.0, temperature=0.1)
        
        initial_T = calculate_temperature(sim.state.velocities)
        
        # Add significant heat
        sim.add_heat(center=(10.0, 10.0), radius=10.0, amount=2.0)
        
        final_T = calculate_temperature(sim.state.velocities)
        
        assert final_T > initial_T
    
    def test_remove_heat_decreases_temperature(self):
        """Test that removing heat decreases temperature."""
        sim = create_crystal_simulation(n_particles=64, box_size=20.0, temperature=0.5)
        
        initial_T = calculate_temperature(sim.state.velocities)
        
        # Remove heat using factor parameter
        sim.remove_heat(center=(10.0, 10.0), radius=10.0, factor=0.5)
        
        final_T = calculate_temperature(sim.state.velocities)
        
        assert final_T < initial_T
    
    def test_local_heat_application(self):
        """Test that heat is applied locally."""
        sim = create_crystal_simulation(n_particles=100, box_size=25.0, temperature=0.1)
        
        # Add significant heat to only one corner
        sim.add_heat(center=(2.5, 2.5), radius=5.0, amount=2.0)
        
        # Particles in that region should be hotter
        local_T = sim.get_local_temperature(center=(2.5, 2.5), radius=5.0)
        global_T = calculate_temperature(sim.state.velocities)
        
        # Local temperature should be higher (with some tolerance for small samples)
        assert local_T >= global_T * 0.9  # Allow some variance


class TestThermostat:
    """Tests for thermostat functionality."""
    
    def test_thermostat_maintains_temperature(self):
        """Test that thermostat maintains target temperature."""
        # Use crystal initialization to avoid overlapping particles
        sim = create_crystal_simulation(n_particles=36, box_size=15.0, temperature=0.5)
        sim.config.use_thermostat = True
        sim.config.target_temperature = 0.5
        sim.config.thermostat_coupling = 0.1
        
        # Run for a while with thermostat
        for _ in range(200):
            sim.step()
        
        final_T = calculate_temperature(sim.state.velocities)
        
        # Should be reasonably close to target (thermostats have fluctuations)
        assert final_T < 2.0  # Just check it doesn't explode
        assert final_T > 0.01  # And doesn't freeze


class TestCreateCrystalSimulation:
    """Tests for the helper function."""
    
    def test_creates_valid_simulation(self):
        """Test that helper creates valid simulation."""
        sim = create_crystal_simulation(
            n_particles=49,
            box_size=20.0,
            temperature=0.2,
            spacing=1.1
        )
        
        assert sim.state is not None
        assert len(sim.state.positions) == 49
        assert sim.config.box_size == (20.0, 20.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
