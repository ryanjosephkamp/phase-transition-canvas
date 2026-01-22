# Week 1 - Project 2: GPU-Accelerated "Phase Transition Canvas"

## Overview

**Week:** 1 (Jan 20 – Jan 27)  
**Theme:** Numerical Integration, Conservation Laws, and Inter-Atomic Potentials  
**Goal:** Prove you understand *how* biophysics simulations calculate movement and *where* they fail.

---

## Project Details

### The "Gap" It Fills
Mastery of **Thermodynamics**, **Statistical Mechanics**, and **GPU Computing**. It demonstrates familiarity with the Lennard-Jones potential (the bread and butter of biophysics) and states of matter.

### The Concept
A 2D particle simulation involving thousands of atoms interacting via Lennard-Jones potentials.
- Unlike a standard "movie," this will be a "canvas." The user uses their mouse to "paint" temperature.
- Clicking and dragging adds kinetic energy (heat) to local particles, melting a solid crystal into a liquid or boiling it into a gas in real-time.

### Novelty/Creative Angle
Standard MD simulations are batch-processed and watched later. This is **real-time computational interaction**. By using **JAX** or **CuPy** to run the physics on your GPU, you can simulate 10x–100x more particles than a standard CPU demo, making the phase transitions look physically realistic rather than jerky.

### Technical Implementation
- **Language:** Python.
- **Libraries:** **JAX** (for differentiable, GPU-accelerated physics operations) or **Numba** (CUDA).
- **Visualization:** **Vispy** or **PyGame** (must handle high frame-rate rendering of thousands of points).

### The "Paper" & Interactive Element
- *Interactive:* A web-embeddable window (potentially exporting the JAX logic to TensorFlow.js or running via a backend) where users observe a crystal lattice and "melt" holes in it with a cursor.
- *Paper Focus:* "Real-Time Visualization of Localized Phase Transitions in Lennard-Jones Fluids using Hardware-Accelerated Dynamics."

---

## Progress Tracking

- [ ] Initial research and planning
- [ ] Core implementation
- [ ] Testing and validation
- [ ] Documentation and paper draft
- [ ] Interactive demo creation
