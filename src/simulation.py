#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Molecular Dynamics Simulation Engine
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         simulation.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

Core molecular dynamics simulation engine using Velocity Verlet integration.
Optimized for real-time performance with interactive temperature painting.
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass, field
import time

from .physics import (
    LennardJonesParameters,
    compute_forces_and_energy,
    build_cell_list,
    compute_forces_cell_list,
    calculate_kinetic_energy,
    calculate_temperature,
    apply_periodic_boundaries
)


@dataclass
class SimulationState:
    """Current state of the simulation."""
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    time: float = 0.0
    step: int = 0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    temperature: float = 0.0
    
    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy
    
    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]


@dataclass
class SimulationConfig:
    """Configuration for the MD simulation."""
    # Box parameters
    box_size: Tuple[float, float] = (30.0, 30.0)
    
    # Time integration
    dt: float = 0.002  # Time step in reduced units
    
    # Lennard-Jones parameters
    lj_params: LennardJonesParameters = field(default_factory=LennardJonesParameters)
    
    # Performance settings
    use_cell_list: bool = True
    max_particles_per_cell: int = 50
    
    # Thermostat
    use_thermostat: bool = False
    target_temperature: float = 1.0
    thermostat_coupling: float = 0.1  # Berendsen coupling constant


class MDSimulation:
    """
    Molecular Dynamics simulation engine.
    
    Uses Velocity Verlet integration for symplectic time evolution.
    Supports periodic boundary conditions and optional thermostats.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.state: Optional[SimulationState] = None
        
        # Performance tracking
        self.steps_per_second = 0.0
        self._last_time = time.time()
        self._step_count = 0
    
    def initialize_crystal(
        self,
        n_particles_x: int = 10,
        n_particles_y: int = 10,
        spacing: float = 1.2,
        temperature: float = 0.1
    ) -> SimulationState:
        """
        Initialize particles in a crystal lattice configuration.
        
        Args:
            n_particles_x: Number of particles in x direction
            n_particles_y: Number of particles in y direction
            spacing: Lattice spacing (in units of sigma)
            temperature: Initial temperature for random velocities
            
        Returns:
            Initial simulation state
        """
        n_particles = n_particles_x * n_particles_y
        
        # Create lattice positions
        positions = np.zeros((n_particles, 2))
        idx = 0
        
        # Center the crystal in the box
        Lx, Ly = self.config.box_size
        offset_x = (Lx - (n_particles_x - 1) * spacing) / 2
        offset_y = (Ly - (n_particles_y - 1) * spacing) / 2
        
        for i in range(n_particles_x):
            for j in range(n_particles_y):
                positions[idx, 0] = offset_x + i * spacing
                positions[idx, 1] = offset_y + j * spacing
                idx += 1
        
        # Initialize random velocities with Maxwell-Boltzmann distribution
        velocities = np.random.randn(n_particles, 2) * np.sqrt(temperature)
        
        # Remove center of mass velocity
        velocities -= np.mean(velocities, axis=0)
        
        # Compute initial forces
        forces, potential_energy = self._compute_forces(positions)
        kinetic_energy = calculate_kinetic_energy(velocities)
        temp = calculate_temperature(velocities)
        
        self.state = SimulationState(
            positions=positions,
            velocities=velocities,
            forces=forces,
            time=0.0,
            step=0,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            temperature=temp
        )
        
        return self.state
    
    def initialize_random(
        self,
        n_particles: int = 100,
        temperature: float = 1.0
    ) -> SimulationState:
        """
        Initialize particles randomly (gas-like configuration).
        
        Args:
            n_particles: Number of particles
            temperature: Initial temperature
            
        Returns:
            Initial simulation state
        """
        Lx, Ly = self.config.box_size
        
        # Random positions
        positions = np.random.rand(n_particles, 2)
        positions[:, 0] *= Lx
        positions[:, 1] *= Ly
        
        # Random velocities
        velocities = np.random.randn(n_particles, 2) * np.sqrt(temperature)
        velocities -= np.mean(velocities, axis=0)
        
        # Compute initial forces
        forces, potential_energy = self._compute_forces(positions)
        kinetic_energy = calculate_kinetic_energy(velocities)
        temp = calculate_temperature(velocities)
        
        self.state = SimulationState(
            positions=positions,
            velocities=velocities,
            forces=forces,
            time=0.0,
            step=0,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            temperature=temp
        )
        
        return self.state
    
    def _compute_forces(self, positions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute forces using appropriate algorithm."""
        lj = self.config.lj_params
        
        if self.config.use_cell_list and positions.shape[0] > 100:
            # Use cell list for large systems
            cell_list, cell_count, n_cells, cell_size = build_cell_list(
                positions,
                self.config.box_size,
                lj.cutoff_distance,
                self.config.max_particles_per_cell
            )
            
            return compute_forces_cell_list(
                positions, cell_list, cell_count,
                n_cells, cell_size, self.config.box_size,
                lj.epsilon, lj.sigma, lj.cutoff,
                self.config.max_particles_per_cell
            )
        else:
            # Direct O(N²) calculation for small systems
            return compute_forces_and_energy(
                positions,
                self.config.box_size,
                lj.epsilon, lj.sigma, lj.cutoff
            )
    
    def step(self) -> SimulationState:
        """
        Perform one Velocity Verlet integration step.
        
        The Velocity Verlet algorithm is:
        1. v(t + dt/2) = v(t) + (dt/2) * a(t)
        2. x(t + dt) = x(t) + dt * v(t + dt/2)
        3. Compute a(t + dt) from x(t + dt)
        4. v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
        
        Returns:
            Updated simulation state
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized")
        
        dt = self.config.dt
        pos = self.state.positions
        vel = self.state.velocities
        forces = self.state.forces
        
        # Step 1: Half-step velocity update
        vel_half = vel + 0.5 * dt * forces
        
        # Step 2: Full position update
        pos_new = pos + dt * vel_half
        
        # Apply periodic boundaries
        pos_new = apply_periodic_boundaries(pos_new, self.config.box_size)
        
        # Step 3: Compute new forces
        forces_new, potential_energy = self._compute_forces(pos_new)
        
        # Step 4: Complete velocity update
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply thermostat if enabled
        if self.config.use_thermostat:
            vel_new = self._apply_berendsen_thermostat(vel_new)
        
        # Compute energies and temperature
        kinetic_energy = calculate_kinetic_energy(vel_new)
        temperature = calculate_temperature(vel_new)
        
        # Update state
        self.state = SimulationState(
            positions=pos_new,
            velocities=vel_new,
            forces=forces_new,
            time=self.state.time + dt,
            step=self.state.step + 1,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            temperature=temperature
        )
        
        # Track performance
        self._step_count += 1
        if self._step_count % 100 == 0:
            current_time = time.time()
            elapsed = current_time - self._last_time
            if elapsed > 0:
                self.steps_per_second = 100.0 / elapsed
            self._last_time = current_time
        
        return self.state
    
    def run(self, n_steps: int) -> SimulationState:
        """Run simulation for n_steps."""
        for _ in range(n_steps):
            self.step()
        return self.state
    
    def _apply_berendsen_thermostat(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply Berendsen thermostat for temperature control.
        
        Scales velocities to approach target temperature.
        Not rigorous (doesn't sample canonical ensemble) but simple and stable.
        """
        current_temp = calculate_temperature(velocities)
        
        if current_temp < 1e-10:
            return velocities
        
        target = self.config.target_temperature
        tau = self.config.thermostat_coupling
        dt = self.config.dt
        
        # Berendsen scaling factor
        lambda_scale = np.sqrt(1.0 + (dt / tau) * (target / current_temp - 1.0))
        
        return velocities * lambda_scale
    
    def add_heat(
        self,
        center: Tuple[float, float],
        radius: float,
        amount: float
    ) -> None:
        """
        Add kinetic energy (heat) to particles near a point.
        
        This is the core of the "temperature painting" feature.
        
        Args:
            center: (x, y) coordinates of heat source
            radius: Radius of effect
            amount: Amount of kinetic energy to add per particle
        """
        if self.state is None:
            return
        
        cx, cy = center
        
        for i in range(self.state.n_particles):
            px, py = self.state.positions[i]
            
            # Distance to heat source (with PBC)
            dx = px - cx
            dy = py - cy
            
            Lx, Ly = self.config.box_size
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < radius:
                # Add random velocity kick proportional to closeness
                factor = 1.0 - dist / radius
                kick = amount * factor
                
                # Random direction
                angle = np.random.rand() * 2 * np.pi
                self.state.velocities[i, 0] += kick * np.cos(angle)
                self.state.velocities[i, 1] += kick * np.sin(angle)
    
    def remove_heat(
        self,
        center: Tuple[float, float],
        radius: float,
        factor: float = 0.9
    ) -> None:
        """
        Remove kinetic energy (cool) particles near a point.
        
        Args:
            center: (x, y) coordinates of cooling point
            radius: Radius of effect
            factor: Velocity scaling factor (< 1 to cool)
        """
        if self.state is None:
            return
        
        cx, cy = center
        
        for i in range(self.state.n_particles):
            px, py = self.state.positions[i]
            
            # Distance to cooling source (with PBC)
            dx = px - cx
            dy = py - cy
            
            Lx, Ly = self.config.box_size
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < radius:
                # Scale down velocity
                scale = factor + (1.0 - factor) * (dist / radius)
                self.state.velocities[i] *= scale
    
    def get_local_temperature(
        self,
        center: Tuple[float, float],
        radius: float
    ) -> float:
        """
        Calculate temperature of particles in a local region.
        
        Args:
            center: (x, y) coordinates of region center
            radius: Region radius
            
        Returns:
            Local temperature
        """
        if self.state is None:
            return 0.0
        
        cx, cy = center
        local_velocities = []
        
        for i in range(self.state.n_particles):
            px, py = self.state.positions[i]
            
            dx = px - cx
            dy = py - cy
            
            Lx, Ly = self.config.box_size
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < radius:
                local_velocities.append(self.state.velocities[i])
        
        if len(local_velocities) < 2:
            return self.state.temperature
        
        return calculate_temperature(np.array(local_velocities))


def create_crystal_simulation(
    n_particles: int = 100,
    box_size: float = 20.0,
    temperature: float = 0.1,
    spacing: float = 1.2
) -> MDSimulation:
    """
    Create a simulation with particles in a crystal lattice.
    
    Args:
        n_particles: Approximate number of particles (will be adjusted to fit grid)
        box_size: Size of simulation box (square)
        temperature: Initial temperature
        spacing: Lattice spacing
        
    Returns:
        Initialized MDSimulation
    """
    # Calculate grid dimensions
    n_side = int(np.sqrt(n_particles))
    n_particles = n_side * n_side  # Adjust to perfect square
    
    config = SimulationConfig(
        box_size=(box_size, box_size),
        dt=0.002,
        use_cell_list=True
    )
    
    sim = MDSimulation(config)
    sim.initialize_crystal(n_side, n_side, spacing, temperature)
    
    return sim
