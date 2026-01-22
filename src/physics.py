#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Lennard-Jones Physics Engine
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         physics.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

This module implements the Lennard-Jones potential and force calculations
for molecular dynamics simulations. The Lennard-Jones potential is the
fundamental inter-atomic potential used in biophysics simulations.

The potential is:
    V(r) = 4ε [(σ/r)¹² - (σ/r)⁶]

Where:
    - ε (epsilon): Depth of the potential well (energy scale)
    - σ (sigma): Distance at which potential is zero (length scale)
    - r: Distance between two particles

The r⁻¹² term models short-range repulsion (Pauli exclusion)
The r⁻⁶ term models long-range attraction (van der Waals/London dispersion)
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LennardJonesParameters:
    """
    Parameters for the Lennard-Jones potential.
    
    Default values are in reduced units where:
    - σ = 1 (length unit)
    - ε = 1 (energy unit)
    - m = 1 (mass unit)
    
    This gives time unit τ = σ√(m/ε)
    """
    epsilon: float = 1.0      # Potential well depth
    sigma: float = 1.0        # Zero-crossing distance
    cutoff: float = 2.5       # Cutoff distance (in units of sigma)
    
    @property
    def r_min(self) -> float:
        """Distance at potential minimum: r_min = 2^(1/6) * σ ≈ 1.122σ"""
        return self.sigma * (2.0 ** (1.0/6.0))
    
    @property
    def cutoff_distance(self) -> float:
        """Actual cutoff distance in length units."""
        return self.cutoff * self.sigma


@jit(nopython=True, cache=True)
def lennard_jones_potential(r: float, epsilon: float, sigma: float) -> float:
    """
    Calculate the Lennard-Jones potential energy.
    
    V(r) = 4ε [(σ/r)¹² - (σ/r)⁶]
    
    Args:
        r: Distance between particles
        epsilon: Potential well depth
        sigma: Zero-crossing distance
        
    Returns:
        Potential energy
    """
    if r < 1e-10:
        return 1e10  # Avoid division by zero
    
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)


@jit(nopython=True, cache=True)
def lennard_jones_force_magnitude(r: float, epsilon: float, sigma: float) -> float:
    """
    Calculate the magnitude of the Lennard-Jones force.
    
    F(r) = -dV/dr = 24ε/r [2(σ/r)¹² - (σ/r)⁶]
    
    Positive values indicate repulsion, negative indicate attraction.
    
    Args:
        r: Distance between particles
        epsilon: Potential well depth
        sigma: Zero-crossing distance
        
    Returns:
        Force magnitude (positive = repulsive)
    """
    if r < 1e-10:
        return 1e10  # Strong repulsion at very close range
    
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return 24.0 * epsilon / r * (2.0 * sr12 - sr6)


@jit(nopython=True, parallel=True, cache=True)
def compute_forces_and_energy(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    epsilon: float,
    sigma: float,
    cutoff: float
) -> Tuple[np.ndarray, float]:
    """
    Compute all pairwise forces and total potential energy.
    
    Uses periodic boundary conditions and a cutoff distance for efficiency.
    Parallelized with Numba for performance.
    
    Args:
        positions: Nx2 array of particle positions
        box_size: (Lx, Ly) simulation box dimensions
        epsilon: LJ epsilon parameter
        sigma: LJ sigma parameter
        cutoff: Cutoff distance (in sigma units)
        
    Returns:
        forces: Nx2 array of forces on each particle
        potential_energy: Total potential energy
    """
    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    
    cutoff_dist = cutoff * sigma
    cutoff_sq = cutoff_dist * cutoff_dist
    
    Lx, Ly = box_size
    
    # Pairwise force calculation
    for i in prange(n_particles):
        for j in range(i + 1, n_particles):
            # Displacement vector with periodic boundary conditions
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            # Minimum image convention
            if dx > Lx / 2:
                dx -= Lx
            elif dx < -Lx / 2:
                dx += Lx
                
            if dy > Ly / 2:
                dy -= Ly
            elif dy < -Ly / 2:
                dy += Ly
            
            r_sq = dx * dx + dy * dy
            
            # Apply cutoff
            if r_sq < cutoff_sq and r_sq > 1e-10:
                r = np.sqrt(r_sq)
                
                # Force magnitude
                f_mag = lennard_jones_force_magnitude(r, epsilon, sigma)
                
                # Force components
                fx = f_mag * dx / r
                fy = f_mag * dy / r
                
                # Newton's third law
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
                
                # Potential energy (count each pair once)
                potential_energy += lennard_jones_potential(r, epsilon, sigma)
    
    return forces, potential_energy


@jit(nopython=True, cache=True)
def compute_forces_cell_list(
    positions: np.ndarray,
    cell_list: np.ndarray,
    cell_count: np.ndarray,
    n_cells: Tuple[int, int],
    cell_size: Tuple[float, float],
    box_size: Tuple[float, float],
    epsilon: float,
    sigma: float,
    cutoff: float,
    max_particles_per_cell: int
) -> Tuple[np.ndarray, float]:
    """
    Compute forces using cell lists for O(N) scaling.
    
    This is much faster for large systems where cutoff << box_size.
    """
    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    
    cutoff_dist = cutoff * sigma
    cutoff_sq = cutoff_dist * cutoff_dist
    
    Lx, Ly = box_size
    nx, ny = n_cells
    cx, cy = cell_size
    
    # For each particle, check only neighboring cells
    for i in range(n_particles):
        # Find cell of particle i
        ci_x = int(positions[i, 0] / cx) % nx
        ci_y = int(positions[i, 1] / cy) % ny
        
        # Loop over neighboring cells (including own cell)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                cj_x = (ci_x + di) % nx
                cj_y = (ci_y + dj) % ny
                
                cell_idx = cj_x * ny + cj_y
                n_in_cell = cell_count[cell_idx]
                
                for k in range(n_in_cell):
                    j = cell_list[cell_idx, k]
                    
                    if j <= i:
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
                        r = np.sqrt(r_sq)
                        f_mag = lennard_jones_force_magnitude(r, epsilon, sigma)
                        
                        fx = f_mag * dx / r
                        fy = f_mag * dy / r
                        
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        
                        potential_energy += lennard_jones_potential(r, epsilon, sigma)
    
    return forces, potential_energy


def build_cell_list(
    positions: np.ndarray,
    box_size: Tuple[float, float],
    cutoff: float,
    max_particles_per_cell: int = 50
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[float, float]]:
    """
    Build cell list data structure for efficient neighbor finding.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) box dimensions
        cutoff: Cutoff distance
        max_particles_per_cell: Maximum particles per cell
        
    Returns:
        cell_list: 2D array [n_cells, max_per_cell] of particle indices
        cell_count: 1D array [n_cells] of particle counts per cell
        n_cells: (nx, ny) number of cells in each direction
        cell_size: (cx, cy) size of each cell
    """
    Lx, Ly = box_size
    
    # Cell size should be >= cutoff for correctness
    nx = max(1, int(Lx / cutoff))
    ny = max(1, int(Ly / cutoff))
    
    cx = Lx / nx
    cy = Ly / ny
    
    n_cells_total = nx * ny
    
    cell_list = np.zeros((n_cells_total, max_particles_per_cell), dtype=np.int32)
    cell_count = np.zeros(n_cells_total, dtype=np.int32)
    
    for i, pos in enumerate(positions):
        ci_x = int(pos[0] / cx) % nx
        ci_y = int(pos[1] / cy) % ny
        cell_idx = ci_x * ny + ci_y
        
        count = cell_count[cell_idx]
        if count < max_particles_per_cell:
            cell_list[cell_idx, count] = i
            cell_count[cell_idx] += 1
    
    return cell_list, cell_count, (nx, ny), (cx, cy)


def calculate_kinetic_energy(velocities: np.ndarray, masses: Optional[np.ndarray] = None) -> float:
    """
    Calculate total kinetic energy.
    
    KE = Σ (1/2) m v²
    
    Args:
        velocities: Nx2 array of velocities
        masses: Optional array of masses (default: all 1.0)
        
    Returns:
        Total kinetic energy
    """
    if masses is None:
        masses = np.ones(velocities.shape[0])
    
    v_sq = np.sum(velocities ** 2, axis=1)
    return 0.5 * np.sum(masses * v_sq)


def calculate_temperature(velocities: np.ndarray, n_dof: Optional[int] = None) -> float:
    """
    Calculate temperature from velocities using equipartition theorem.
    
    In 2D: T = (2 * KE) / (N * k_B * 2) = KE / N
    
    In reduced units where k_B = 1 and m = 1:
    T = <v²> / 2  (per degree of freedom)
    
    Args:
        velocities: Nx2 array of velocities
        n_dof: Number of degrees of freedom (default: 2N - 2 for 2D with fixed COM)
        
    Returns:
        Temperature in reduced units
    """
    n_particles = velocities.shape[0]
    
    if n_dof is None:
        n_dof = 2 * n_particles - 2  # Remove COM translation
    
    if n_dof <= 0:
        return 0.0
    
    kinetic_energy = calculate_kinetic_energy(velocities)
    return 2.0 * kinetic_energy / n_dof


def apply_periodic_boundaries(positions: np.ndarray, box_size: Tuple[float, float]) -> np.ndarray:
    """
    Apply periodic boundary conditions to positions.
    
    Args:
        positions: Nx2 array of positions
        box_size: (Lx, Ly) simulation box dimensions
        
    Returns:
        Wrapped positions
    """
    Lx, Ly = box_size
    positions[:, 0] = positions[:, 0] % Lx
    positions[:, 1] = positions[:, 1] % Ly
    return positions


def minimum_image_distance(
    pos1: np.ndarray, 
    pos2: np.ndarray, 
    box_size: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Calculate the minimum image distance between two particles.
    
    Uses the minimum image convention for periodic boundary conditions.
    
    Args:
        pos1: Position of first particle (x, y)
        pos2: Position of second particle (x, y)
        box_size: (Lx, Ly) simulation box dimensions
        
    Returns:
        (dx, dy, r): Displacement components and distance
    """
    Lx, Ly = box_size
    
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    # Minimum image convention
    if dx > Lx / 2:
        dx -= Lx
    elif dx < -Lx / 2:
        dx += Lx
        
    if dy > Ly / 2:
        dy -= Ly
    elif dy < -Ly / 2:
        dy += Ly
    
    r = np.sqrt(dx * dx + dy * dy)
    
    return dx, dy, r
