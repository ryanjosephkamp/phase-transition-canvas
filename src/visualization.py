#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Real-Time Visualization Module
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         visualization.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

This module provides visualization tools for the molecular dynamics simulation:
- Color mapping based on particle temperature/velocity
- Particle rendering for Matplotlib and Streamlit
- Temperature field visualization
- Phase diagram overlays
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, EllipseCollection
import matplotlib.animation as animation
from typing import Tuple, Optional, List
from dataclasses import dataclass
import io


# Custom colormaps
def create_temperature_colormap():
    """
    Create a colormap for temperature visualization.
    
    Blue (cold) -> Cyan -> Green -> Yellow -> Red (hot)
    """
    colors = [
        (0.0, 0.0, 0.5),    # Dark blue (very cold)
        (0.0, 0.5, 1.0),    # Light blue
        (0.0, 1.0, 1.0),    # Cyan
        (0.0, 1.0, 0.0),    # Green
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.5, 0.0),    # Orange
        (1.0, 0.0, 0.0),    # Red (hot)
    ]
    return LinearSegmentedColormap.from_list("temperature", colors, N=256)


def create_phase_colormap():
    """
    Create a colormap for phase visualization.
    
    Blue (solid) -> Purple (liquid) -> Red (gas)
    """
    colors = [
        (0.2, 0.4, 0.8),    # Blue (solid)
        (0.6, 0.2, 0.8),    # Purple (liquid)
        (1.0, 0.3, 0.3),    # Red (gas)
    ]
    return LinearSegmentedColormap.from_list("phase", colors, N=256)


TEMPERATURE_CMAP = create_temperature_colormap()
PHASE_CMAP = create_phase_colormap()


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    particle_radius: float = 0.4
    show_velocities: bool = False
    velocity_scale: float = 0.3
    color_by: str = "temperature"  # "temperature", "speed", "order", "phase"
    show_trails: bool = False
    trail_length: int = 10
    background_color: str = "#1a1a2e"
    show_periodic_images: bool = False
    min_temperature: float = 0.0
    max_temperature: float = 2.0
    figsize: Tuple[int, int] = (8, 8)


def calculate_particle_colors(
    velocities: np.ndarray,
    config: VisualizationConfig,
    order_params: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate colors for particles based on configuration.
    
    Args:
        velocities: Nx2 array of velocities
        config: Visualization configuration
        order_params: Optional array of order parameters
        
    Returns:
        Nx4 RGBA color array
    """
    n_particles = len(velocities)
    
    # Calculate speed (related to local temperature)
    speeds = np.sqrt(np.sum(velocities**2, axis=1))
    
    if config.color_by == "speed":
        # Color by speed directly
        norm_speeds = np.clip(speeds / 3.0, 0, 1)
        colors = TEMPERATURE_CMAP(norm_speeds)
        
    elif config.color_by == "temperature":
        # Color by kinetic temperature (T = v²/2 for m=1, kB=1)
        temperatures = speeds**2 / 2
        norm_temps = np.clip(
            (temperatures - config.min_temperature) / 
            (config.max_temperature - config.min_temperature),
            0, 1
        )
        colors = TEMPERATURE_CMAP(norm_temps)
        
    elif config.color_by == "order" and order_params is not None:
        # Color by order parameter
        colors = PHASE_CMAP(1 - order_params)  # High order = blue, low = red
        
    else:
        # Default: uniform color
        colors = np.zeros((n_particles, 4))
        colors[:, 0] = 0.3  # R
        colors[:, 1] = 0.6  # G
        colors[:, 2] = 0.9  # B
        colors[:, 3] = 1.0  # A
    
    return colors


def render_particles_matplotlib(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: Tuple[float, float],
    config: Optional[VisualizationConfig] = None,
    ax: Optional[plt.Axes] = None,
    order_params: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Render particles using Matplotlib.
    
    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities
        box_size: (Lx, Ly) simulation box dimensions
        config: Visualization configuration
        ax: Optional existing axes to draw on
        order_params: Optional order parameters for coloring
        
    Returns:
        Matplotlib figure
    """
    if config is None:
        config = VisualizationConfig()
    
    Lx, Ly = box_size
    
    # Create figure with no extra spacing
    if ax is None:
        fig = plt.figure(figsize=config.figsize)
        ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    else:
        fig = ax.figure
    
    ax.clear()
    ax.set_facecolor(config.background_color)
    fig.patch.set_facecolor(config.background_color)
    
    # Calculate colors
    colors = calculate_particle_colors(velocities, config, order_params)
    
    # Draw particles using scatter for efficiency
    # Scale particle size relative to box - larger box means smaller relative particles
    base_size = 300  # Base marker size for box_size=20
    scale_factor = (20.0 / max(Lx, Ly)) ** 2
    sizes = base_size * scale_factor * config.particle_radius
    
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=sizes,
        c=colors,
        edgecolors='white',
        linewidths=0.3,
        alpha=0.9
    )
    
    # Draw velocity arrows if enabled - use quiver for better scaling
    if config.show_velocities and len(positions) > 0:
        # Scale arrows relative to box size
        arrow_scale = 15.0 / max(Lx, Ly)  # Inverse scale - larger values = shorter arrows
        ax.quiver(
            positions[:, 0], positions[:, 1],
            velocities[:, 0], velocities[:, 1],
            color='white', alpha=0.5,
            scale=arrow_scale * 10,
            scale_units='width',
            width=0.003,
            headwidth=4,
            headlength=5
        )
    
    # Set axis limits with small margin
    margin = max(Lx, Ly) * 0.02
    ax.set_xlim(-margin, Lx + margin)
    ax.set_ylim(-margin, Ly + margin)
    ax.set_aspect('equal')
    
    # Draw box boundary
    ax.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], 
            'white', linewidth=1.5, alpha=0.5)
    
    # Remove all decorations
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    return fig


def render_particles_streamlit(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: Tuple[float, float],
    config: Optional[VisualizationConfig] = None,
    order_params: Optional[np.ndarray] = None
) -> bytes:
    """
    Render particles and return PNG bytes for Streamlit.
    
    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities
        box_size: (Lx, Ly) simulation box dimensions
        config: Visualization configuration
        order_params: Optional order parameters
        
    Returns:
        PNG image as bytes
    """
    fig = render_particles_matplotlib(
        positions, velocities, box_size, config, order_params=order_params
    )
    
    buf = io.BytesIO()
    # Save with no extra padding - figure already has no margins
    fig.savefig(buf, format='png', dpi=100, 
                facecolor=fig.get_facecolor(), edgecolor='none',
                pad_inches=0, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def render_temperature_field(
    temperature_field: np.ndarray,
    box_size: Tuple[float, float],
    config: Optional[VisualizationConfig] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Render the temperature field as a heatmap.
    
    Args:
        temperature_field: 2D array of temperatures
        box_size: (Lx, Ly) simulation box dimensions
        config: Visualization configuration
        ax: Optional existing axes
        
    Returns:
        Matplotlib figure
    """
    if config is None:
        config = VisualizationConfig()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    else:
        fig = ax.figure
    
    ax.clear()
    
    extent = [0, box_size[0], 0, box_size[1]]
    im = ax.imshow(
        temperature_field.T,  # Transpose for correct orientation
        origin='lower',
        extent=extent,
        cmap=TEMPERATURE_CMAP,
        vmin=config.min_temperature,
        vmax=config.max_temperature,
        aspect='equal',
        interpolation='bilinear'
    )
    
    plt.colorbar(im, ax=ax, label='Temperature')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Temperature Field')
    
    return fig


def render_energy_plot(
    times: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Render energy vs time plot.
    
    Args:
        times: Time array
        kinetic: Kinetic energy array
        potential: Potential energy array
        total: Total energy array
        ax: Optional existing axes
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        fig = ax.figure
    
    ax.clear()
    ax.plot(times, kinetic, 'r-', label='Kinetic', linewidth=1.5)
    ax.plot(times, potential, 'b-', label='Potential', linewidth=1.5)
    ax.plot(times, total, 'k-', label='Total', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig


def render_phase_diagram(
    temperature_history: List[float],
    density_history: List[float],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Render the system trajectory on a simplified phase diagram.
    
    Args:
        temperature_history: List of temperature values
        density_history: List of density values
        ax: Optional existing axes
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Draw approximate phase regions
    T = np.linspace(0, 2, 100)
    
    # Solid-liquid boundary (approximate)
    ax.axhline(y=0.4, color='blue', linestyle='--', alpha=0.5, label='Melting line')
    
    # Liquid-gas boundary (approximate)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Boiling line')
    
    # Fill regions
    ax.axhspan(0, 0.4, alpha=0.1, color='blue', label='Solid')
    ax.axhspan(0.4, 0.8, alpha=0.1, color='green', label='Liquid')
    ax.axhspan(0.8, 2, alpha=0.1, color='red', label='Gas')
    
    # Plot trajectory
    if len(temperature_history) > 0:
        ax.plot(density_history, temperature_history, 'ko-', 
                markersize=2, linewidth=1, alpha=0.7)
        ax.plot(density_history[-1], temperature_history[-1], 
                'ko', markersize=10, label='Current')
    
    ax.set_xlabel('Density (ρ)')
    ax.set_ylabel('Temperature (T)')
    ax.set_title('Phase Diagram')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 2)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return fig


def create_animation(
    simulation_history: List[Tuple[np.ndarray, np.ndarray]],
    box_size: Tuple[float, float],
    config: Optional[VisualizationConfig] = None,
    fps: int = 30
) -> animation.FuncAnimation:
    """
    Create an animation from simulation history.
    
    Args:
        simulation_history: List of (positions, velocities) tuples
        box_size: (Lx, Ly) simulation box dimensions
        config: Visualization configuration
        fps: Frames per second
        
    Returns:
        Matplotlib animation
    """
    if config is None:
        config = VisualizationConfig()
    
    fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    
    def update(frame):
        positions, velocities = simulation_history[frame]
        render_particles_matplotlib(positions, velocities, box_size, config, ax=ax)
        return ax,
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(simulation_history),
        interval=1000/fps, blit=True
    )
    
    return ani


class ParticleTrail:
    """Track particle positions for trail visualization."""
    
    def __init__(self, n_particles: int, trail_length: int = 10):
        self.n_particles = n_particles
        self.trail_length = trail_length
        self.history: List[np.ndarray] = []
    
    def update(self, positions: np.ndarray):
        """Add new positions to trail."""
        self.history.append(positions.copy())
        if len(self.history) > self.trail_length:
            self.history.pop(0)
    
    def get_trails(self) -> List[np.ndarray]:
        """Get trail data for visualization."""
        return self.history
    
    def clear(self):
        """Clear trail history."""
        self.history = []


def draw_heat_brush_preview(
    ax: plt.Axes,
    center: Tuple[float, float],
    radius: float,
    heating: bool = True
):
    """
    Draw a preview of the heat brush.
    
    Args:
        ax: Matplotlib axes
        center: (x, y) brush center
        radius: Brush radius
        heating: True for heating (red), False for cooling (blue)
    """
    color = 'red' if heating else 'blue'
    circle = plt.Circle(center, radius, fill=False, color=color, 
                       linewidth=2, linestyle='--', alpha=0.7)
    ax.add_patch(circle)


def get_phase_indicator_color(phase_name: str) -> str:
    """Get color for phase indicator."""
    colors = {
        'solid': '#3b82f6',   # Blue
        'liquid': '#8b5cf6',  # Purple
        'gas': '#ef4444',     # Red
        'unknown': '#6b7280'  # Gray
    }
    return colors.get(phase_name.lower(), colors['unknown'])


def render_dashboard(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: Tuple[float, float],
    energy_history: dict,
    temperature_history: List[float],
    config: Optional[VisualizationConfig] = None
) -> plt.Figure:
    """
    Render a complete dashboard with particles and graphs.
    
    Args:
        positions: Nx2 array of positions
        velocities: Nx2 array of velocities
        box_size: (Lx, Ly) simulation box dimensions
        energy_history: Dict with 'time', 'kinetic', 'potential', 'total'
        temperature_history: List of temperature values
        config: Visualization configuration
        
    Returns:
        Matplotlib figure
    """
    if config is None:
        config = VisualizationConfig()
    
    fig = plt.figure(figsize=(14, 8))
    
    # Particle display (left, large)
    ax_particles = fig.add_subplot(2, 2, 1)
    render_particles_matplotlib(positions, velocities, box_size, config, ax=ax_particles)
    ax_particles.set_title('Particle System', color='white')
    
    # Energy plot (top right)
    ax_energy = fig.add_subplot(2, 2, 2)
    if len(energy_history.get('time', [])) > 0:
        render_energy_plot(
            np.array(energy_history['time']),
            np.array(energy_history['kinetic']),
            np.array(energy_history['potential']),
            np.array(energy_history['total']),
            ax=ax_energy
        )
    
    # Temperature plot (bottom right)
    ax_temp = fig.add_subplot(2, 2, 4)
    if len(temperature_history) > 0:
        ax_temp.plot(temperature_history, 'r-', linewidth=1.5)
        ax_temp.set_xlabel('Step')
        ax_temp.set_ylabel('Temperature')
        ax_temp.set_title('Temperature History')
        ax_temp.grid(True, alpha=0.3)
    
    fig.patch.set_facecolor(config.background_color)
    plt.tight_layout()
    
    return fig
