#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GPU-Accelerated Phase Transition Canvas
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Description:    Real-time 2D Lennard-Jones particle simulation with interactive
                temperature painting for visualizing phase transitions

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

This package implements a real-time molecular dynamics simulation featuring:
- Lennard-Jones potential for inter-atomic interactions
- GPU acceleration via NumPy/Numba for performance
- Interactive temperature "painting" with mouse input
- Visualization of solid, liquid, and gas phase transitions

Modules:
    - physics: Lennard-Jones potential and force calculations
    - simulation: Core MD simulation engine with velocity Verlet
    - thermodynamics: Temperature control and phase detection
    - visualization: Real-time rendering of particle states
"""

__version__ = "1.0.0"
__author__ = "Ryan Kamp"
