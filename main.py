#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Phase Transition Canvas - Command Line Interface
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         main.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

Command line interface for running and testing the Phase Transition Canvas
molecular dynamics simulation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import Optional

from src.physics import LennardJonesParameters, calculate_temperature
from src.simulation import MDSimulation, SimulationConfig, create_crystal_simulation
from src.thermodynamics import (
    Phase, identify_phase, calculate_global_order_parameter
)
from src.visualization import (
    VisualizationConfig, render_particles_matplotlib, render_energy_plot
)


def run_equilibration_test(n_particles: int = 100, n_steps: int = 5000):
    """
    Run an equilibration test to verify the simulation is working.
    
    Args:
        n_particles: Number of particles
        n_steps: Number of simulation steps
    """
    print("=" * 60)
    print("Phase Transition Canvas - Equilibration Test")
    print("=" * 60)
    
    # Create simulation
    print(f"\nInitializing {n_particles} particles in crystal configuration...")
    sim = create_crystal_simulation(
        n_particles=n_particles,
        box_size=20.0,
        temperature=0.2,
        spacing=1.1
    )
    
    # Run simulation
    print(f"Running {n_steps} steps...")
    
    temperatures = []
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    times = []
    
    t_start = time.time()
    
    for step in range(n_steps):
        sim.step()
        
        if step % 100 == 0:
            state = sim.state
            temp = calculate_temperature(state.velocities)
            
            temperatures.append(temp)
            kinetic_energies.append(state.kinetic_energy)
            potential_energies.append(state.potential_energy)
            total_energies.append(state.total_energy)
            times.append(step * sim.config.dt)
            
            if step % 1000 == 0:
                print(f"  Step {step:5d}: T = {temp:.4f}, E = {state.total_energy:.2f}")
    
    t_end = time.time()
    
    print(f"\nSimulation completed in {t_end - t_start:.2f} seconds")
    print(f"Steps per second: {n_steps / (t_end - t_start):.1f}")
    
    # Calculate final properties
    state = sim.state
    final_temp = calculate_temperature(state.velocities)
    order_param = calculate_global_order_parameter(
        state.positions, sim.config.box_size
    )
    density = n_particles / (sim.config.box_size[0] * sim.config.box_size[1])
    phase_info = identify_phase(final_temp, density, order_param)
    
    print(f"\nFinal State:")
    print(f"  Temperature:     {final_temp:.4f}")
    print(f"  Order Parameter: {order_param:.4f}")
    print(f"  Phase:           {phase_info.phase.value}")
    print(f"  Kinetic Energy:  {state.kinetic_energy:.2f}")
    print(f"  Potential Energy:{state.potential_energy:.2f}")
    print(f"  Total Energy:    {state.total_energy:.2f}")
    
    # Energy conservation check
    energy_drift = abs(total_energies[-1] - total_energies[0]) / abs(total_energies[0])
    print(f"\n  Energy drift:    {energy_drift * 100:.4f}%")
    
    if energy_drift < 0.01:
        print("  ✓ Excellent energy conservation!")
    elif energy_drift < 0.05:
        print("  ✓ Good energy conservation")
    else:
        print("  ⚠ Consider using smaller timestep")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy plot
    ax = axes[0, 0]
    ax.plot(times, kinetic_energies, 'r-', label='Kinetic')
    ax.plot(times, potential_energies, 'b-', label='Potential')
    ax.plot(times, total_energies, 'k-', label='Total', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature plot
    ax = axes[0, 1]
    ax.plot(times, temperatures, 'r-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature vs Time')
    ax.grid(True, alpha=0.3)
    
    # Final configuration
    ax = axes[1, 0]
    config = VisualizationConfig()
    render_particles_matplotlib(
        state.positions, state.velocities, sim.config.box_size, config, ax=ax
    )
    ax.set_title('Final Configuration')
    
    # Energy histogram
    ax = axes[1, 1]
    speeds = np.linalg.norm(state.velocities, axis=1)
    ax.hist(speeds, bins=30, density=True, alpha=0.7, color='blue')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Probability Density')
    ax.set_title('Speed Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('equilibration_test.png', dpi=150)
    print(f"\nPlot saved to equilibration_test.png")
    plt.show()


def run_phase_transition_demo(n_particles: int = 100, n_steps: int = 10000):
    """
    Demonstrate phase transition by heating a solid.
    
    Args:
        n_particles: Number of particles
        n_steps: Number of simulation steps
    """
    print("=" * 60)
    print("Phase Transition Canvas - Melting Demonstration")
    print("=" * 60)
    
    # Create cold crystal
    print(f"\nCreating cold crystal with {n_particles} particles...")
    sim = create_crystal_simulation(
        n_particles=n_particles,
        box_size=20.0,
        temperature=0.1,  # Very cold
        spacing=1.1
    )
    
    # Enable thermostat for controlled heating
    sim.config.use_thermostat = True
    sim.config.thermostat_tau = 1.0
    
    temperatures = []
    order_params = []
    phases = []
    
    # Run simulation with gradual heating
    print("\nHeating the crystal gradually...")
    
    target_temps = np.linspace(0.1, 1.5, 50)  # Heat from 0.1 to 1.5
    steps_per_temp = n_steps // len(target_temps)
    
    for i, target_T in enumerate(target_temps):
        sim.config.target_temperature = target_T
        
        for _ in range(steps_per_temp):
            sim.step()
        
        state = sim.state
        actual_T = calculate_temperature(state.velocities)
        
        # Calculate order parameter (expensive, so only periodically)
        order = calculate_global_order_parameter(
            state.positions, sim.config.box_size
        )
        
        density = n_particles / (sim.config.box_size[0] * sim.config.box_size[1])
        phase_info = identify_phase(actual_T, density, order)
        
        temperatures.append(actual_T)
        order_params.append(order)
        phases.append(phase_info.phase.value)
        
        if i % 10 == 0:
            print(f"  T = {actual_T:.3f}, ψ₆ = {order:.3f}, Phase: {phase_info.phase.value}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Order parameter vs temperature
    ax = axes[0]
    ax.plot(temperatures, order_params, 'b.-')
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Solid threshold')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Order Parameter (ψ₆)')
    ax.set_title('Melting Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase diagram
    ax = axes[1]
    phase_colors = {'solid': 'blue', 'liquid': 'purple', 'gas': 'red'}
    colors = [phase_colors.get(p, 'gray') for p in phases]
    ax.scatter(range(len(phases)), temperatures, c=colors, s=50)
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.set_title('Phase Evolution')
    ax.grid(True, alpha=0.3)
    
    # Final configuration
    ax = axes[2]
    config = VisualizationConfig()
    state = sim.state
    render_particles_matplotlib(
        state.positions, state.velocities, sim.config.box_size, config, ax=ax
    )
    ax.set_title(f'Final State (T={temperatures[-1]:.2f})')
    
    plt.tight_layout()
    plt.savefig('phase_transition_demo.png', dpi=150)
    print(f"\nPlot saved to phase_transition_demo.png")
    plt.show()


def run_animation(n_particles: int = 100, n_frames: int = 200, n_steps_per_frame: int = 10):
    """
    Create an animation of the simulation.
    
    Args:
        n_particles: Number of particles
        n_frames: Number of animation frames
        n_steps_per_frame: Simulation steps between frames
    """
    print("=" * 60)
    print("Phase Transition Canvas - Animation")
    print("=" * 60)
    
    # Create simulation
    print(f"\nInitializing {n_particles} particles...")
    sim = create_crystal_simulation(
        n_particles=n_particles,
        box_size=20.0,
        temperature=0.3,
        spacing=1.1
    )
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8))
    vis_config = VisualizationConfig()
    
    def update(frame):
        # Add heat to center periodically
        if frame % 50 == 25:
            sim.add_heat(center=(10, 10), radius=5, amount=0.5)
        
        # Run simulation steps
        for _ in range(n_steps_per_frame):
            sim.step()
        
        # Render
        state = sim.state
        render_particles_matplotlib(
            state.positions, state.velocities, sim.config.box_size, vis_config, ax=ax
        )
        
        temp = calculate_temperature(state.velocities)
        ax.set_title(f'Frame {frame}, T = {temp:.3f}')
        
        return ax,
    
    print(f"Creating animation with {n_frames} frames...")
    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    
    # Save animation
    print("Saving animation (this may take a while)...")
    ani.save('simulation_animation.gif', writer='pillow', fps=20)
    print("Animation saved to simulation_animation.gif")
    
    plt.show()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Phase Transition Canvas - 2D Lennard-Jones Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test              Run equilibration test
  python main.py --demo              Run phase transition demo
  python main.py --animate           Create animation
  python main.py --app               Launch Streamlit app
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run equilibration test')
    parser.add_argument('--demo', action='store_true',
                       help='Run phase transition demonstration')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation')
    parser.add_argument('--app', action='store_true',
                       help='Launch Streamlit web app')
    parser.add_argument('--particles', '-n', type=int, default=100,
                       help='Number of particles (default: 100)')
    parser.add_argument('--steps', '-s', type=int, default=5000,
                       help='Number of simulation steps (default: 5000)')
    
    args = parser.parse_args()
    
    if args.test:
        run_equilibration_test(n_particles=args.particles, n_steps=args.steps)
    elif args.demo:
        run_phase_transition_demo(n_particles=args.particles, n_steps=args.steps)
    elif args.animate:
        run_animation(n_particles=args.particles)
    elif args.app:
        import subprocess
        print("Launching Streamlit app...")
        subprocess.run(['streamlit', 'run', 'app.py'])
    else:
        # Default: run test
        parser.print_help()
        print("\nNo action specified. Run with --test, --demo, --animate, or --app")


if __name__ == "__main__":
    main()
