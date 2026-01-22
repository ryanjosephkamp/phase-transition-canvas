#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Thermodynamics and Phase Detection
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         thermodynamics.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

This module handles thermodynamic analysis and phase detection for the
Lennard-Jones simulation, including:
- Phase identification (solid, liquid, gas)
- Order parameter calculations
- Temperature field visualization
- Phase transition detection
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from numba import jit


class Phase(Enum):
    """Phases of matter in 2D Lennard-Jones system."""
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    UNKNOWN = "unknown"


@dataclass
class PhaseInfo:
    """Information about the current phase."""
    phase: Phase
    order_parameter: float  # 0 = disordered (gas), 1 = ordered (solid)
    local_density: float
    temperature: float
    description: str


# Phase boundaries in reduced units (approximate)
# These are for 2D LJ, actual values depend on density
SOLID_LIQUID_TEMP = 0.4  # Below this is solid
LIQUID_GAS_TEMP = 0.8    # Above this is gas (at low density)
CRITICAL_TEMP = 1.0      # Critical temperature


def identify_phase(
    temperature: float,
    density: float,
    order_parameter: float
) -> PhaseInfo:
    """
    Identify the phase based on temperature, density, and order.
    
    Uses a simplified phase diagram for 2D Lennard-Jones.
    
    Args:
        temperature: System temperature in reduced units
        density: Number density (particles per unit area)
        order_parameter: Measure of crystalline order (0-1)
        
    Returns:
        PhaseInfo with phase identification and description
    """
    # Low temperature, high order = solid
    if temperature < SOLID_LIQUID_TEMP and order_parameter > 0.6:
        return PhaseInfo(
            phase=Phase.SOLID,
            order_parameter=order_parameter,
            local_density=density,
            temperature=temperature,
            description=f"Solid phase: Low T ({temperature:.2f}), high order ({order_parameter:.2f})"
        )
    
    # High temperature, low density = gas
    if temperature > LIQUID_GAS_TEMP and density < 0.3:
        return PhaseInfo(
            phase=Phase.GAS,
            order_parameter=order_parameter,
            local_density=density,
            temperature=temperature,
            description=f"Gas phase: High T ({temperature:.2f}), low density ({density:.2f})"
        )
    
    # Everything else = liquid
    return PhaseInfo(
        phase=Phase.LIQUID,
        order_parameter=order_parameter,
        local_density=density,
        temperature=temperature,
        description=f"Liquid phase: T={temperature:.2f}, ρ={density:.2f}, ψ={order_parameter:.2f}"
    )


@jit(nopython=True, cache=True)
def calculate_local_order_parameter(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    cutoff: float = 1.5
) -> np.ndarray:
    """
    Calculate hexatic order parameter for each particle.
    
    The hexatic order parameter ψ₆ measures 6-fold symmetry:
    ψ₆ = |⟨exp(6iθ)⟩| where θ is the angle to each neighbor
    
    For a perfect hexagonal lattice, ψ₆ = 1.
    For a disordered liquid/gas, ψ₆ ≈ 0.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) box dimensions
        cutoff: Neighbor cutoff distance
        
    Returns:
        Array of order parameters for each particle
    """
    n_particles = positions.shape[0]
    order_params = np.zeros(n_particles)
    
    Lx, Ly = box_size
    cutoff_sq = cutoff * cutoff
    
    for i in range(n_particles):
        # Sum exp(6i*theta) over neighbors
        sum_real = 0.0
        sum_imag = 0.0
        n_neighbors = 0
        
        for j in range(n_particles):
            if i == j:
                continue
            
            # Displacement with PBC
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            r_sq = dx * dx + dy * dy
            
            if r_sq < cutoff_sq and r_sq > 1e-10:
                # Angle to neighbor
                theta = np.arctan2(dy, dx)
                
                # Accumulate exp(6i*theta)
                sum_real += np.cos(6.0 * theta)
                sum_imag += np.sin(6.0 * theta)
                n_neighbors += 1
        
        if n_neighbors > 0:
            # Magnitude of average
            avg_real = sum_real / n_neighbors
            avg_imag = sum_imag / n_neighbors
            order_params[i] = np.sqrt(avg_real * avg_real + avg_imag * avg_imag)
    
    return order_params


def calculate_global_order_parameter(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    cutoff: float = 1.5
) -> float:
    """
    Calculate the global hexatic order parameter.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) box dimensions
        cutoff: Neighbor cutoff distance
        
    Returns:
        Global order parameter (average of local values)
    """
    local_order = calculate_local_order_parameter(positions, box_size, cutoff)
    return np.mean(local_order)


def calculate_density_field(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    grid_size: Tuple[int, int] = (20, 20)
) -> np.ndarray:
    """
    Calculate local density on a grid.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) box dimensions
        grid_size: (nx, ny) grid dimensions
        
    Returns:
        2D array of local densities
    """
    Lx, Ly = box_size
    nx, ny = grid_size
    
    # Grid cell dimensions
    dx = Lx / nx
    dy = Ly / ny
    cell_area = dx * dy
    
    # Count particles in each cell
    counts = np.zeros((nx, ny))
    
    for pos in positions:
        ix = int(pos[0] / dx) % nx
        iy = int(pos[1] / dy) % ny
        counts[ix, iy] += 1
    
    # Convert to density
    density = counts / cell_area
    
    return density


def calculate_temperature_field(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: Tuple[float, float],
    grid_size: Tuple[int, int] = (20, 20)
) -> np.ndarray:
    """
    Calculate local temperature on a grid.
    
    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities
        box_size: (Lx, Ly) box dimensions
        grid_size: (nx, ny) grid dimensions
        
    Returns:
        2D array of local temperatures
    """
    Lx, Ly = box_size
    nx, ny = grid_size
    
    dx = Lx / nx
    dy = Ly / ny
    
    # Accumulate kinetic energy and count
    kinetic_sums = np.zeros((nx, ny))
    counts = np.zeros((nx, ny))
    
    for i in range(len(positions)):
        ix = int(positions[i, 0] / dx) % nx
        iy = int(positions[i, 1] / dy) % ny
        
        v_sq = velocities[i, 0]**2 + velocities[i, 1]**2
        kinetic_sums[ix, iy] += v_sq
        counts[ix, iy] += 1
    
    # Calculate local temperature (T = <v²>/2 in 2D with m=1, kB=1)
    temperature = np.zeros((nx, ny))
    mask = counts > 0
    temperature[mask] = kinetic_sums[mask] / (2 * counts[mask])
    
    return temperature


def calculate_radial_distribution(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    n_bins: int = 100,
    r_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radial distribution function g(r).
    
    g(r) shows the probability of finding a particle at distance r
    relative to a uniform distribution. Peaks indicate preferred distances.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) box dimensions
        n_bins: Number of histogram bins
        r_max: Maximum distance (default: half box size)
        
    Returns:
        r: Array of radial distances
        g: Radial distribution function
    """
    n_particles = len(positions)
    Lx, Ly = box_size
    
    if r_max is None:
        r_max = min(Lx, Ly) / 2
    
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    
    histogram = np.zeros(n_bins)
    
    # Count pairs at each distance
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            # Minimum image
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            r = np.sqrt(dx * dx + dy * dy)
            
            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < n_bins:
                    histogram[bin_idx] += 2  # Count both i-j and j-i
    
    # Normalize by ideal gas
    box_area = Lx * Ly
    rho = n_particles / box_area
    
    g = np.zeros(n_bins)
    for i in range(n_bins):
        r = r_centers[i]
        shell_area = 2 * np.pi * r * dr
        n_ideal = rho * shell_area * n_particles
        if n_ideal > 0:
            g[i] = histogram[i] / n_ideal
    
    return r_centers, g


class PhaseTransitionTracker:
    """
    Track phase transitions over time.
    
    Monitors order parameter and temperature to detect
    when the system crosses phase boundaries.
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.temperature_history: List[float] = []
        self.order_history: List[float] = []
        self.phase_history: List[Phase] = []
        self.time_history: List[float] = []
        
        self.transition_events: List[Tuple[float, Phase, Phase]] = []
    
    def update(
        self,
        time: float,
        temperature: float,
        order_parameter: float,
        density: float
    ) -> Optional[Tuple[Phase, Phase]]:
        """
        Update tracking with new measurements.
        
        Args:
            time: Simulation time
            temperature: Current temperature
            order_parameter: Current order parameter
            density: Current density
            
        Returns:
            (old_phase, new_phase) if transition detected, else None
        """
        # Identify current phase
        phase_info = identify_phase(temperature, density, order_parameter)
        current_phase = phase_info.phase
        
        # Store history
        self.temperature_history.append(temperature)
        self.order_history.append(order_parameter)
        self.phase_history.append(current_phase)
        self.time_history.append(time)
        
        # Trim history
        if len(self.temperature_history) > self.history_length:
            self.temperature_history.pop(0)
            self.order_history.pop(0)
            self.phase_history.pop(0)
            self.time_history.pop(0)
        
        # Check for transition
        if len(self.phase_history) >= 2:
            old_phase = self.phase_history[-2]
            if old_phase != current_phase:
                self.transition_events.append((time, old_phase, current_phase))
                return (old_phase, current_phase)
        
        return None
    
    def get_recent_transitions(self, n: int = 5) -> List[Tuple[float, Phase, Phase]]:
        """Get the n most recent phase transitions."""
        return self.transition_events[-n:]


def melting_indicator(
    temperature: float,
    order_parameter: float,
    lindemann_ratio: float = 0.1
) -> float:
    """
    Calculate a melting indicator (0 = solid, 1 = melted).
    
    Uses a combination of temperature and order parameter.
    
    Args:
        temperature: System temperature
        order_parameter: Hexatic order parameter
        lindemann_ratio: Lindemann melting criterion threshold
        
    Returns:
        Melting indicator (0-1)
    """
    # Temperature contribution
    temp_factor = np.clip((temperature - 0.3) / 0.2, 0, 1)
    
    # Disorder contribution
    disorder_factor = 1.0 - order_parameter
    
    # Combined indicator
    indicator = 0.5 * temp_factor + 0.5 * disorder_factor
    
    return np.clip(indicator, 0, 1)
