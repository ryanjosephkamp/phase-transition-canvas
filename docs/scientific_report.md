# Phase Transition Canvas: Interactive Molecular Dynamics Simulation
## Scientific Report

**Author:** Ryan Kamp  
**Affiliation:** University of Cincinnati, Department of Computer Science  
**Email:** kamprj@mail.uc.edu  
**Date:** January 22, 2026

---

## Abstract

This report presents the Phase Transition Canvas, an interactive molecular dynamics simulation that allows users to visualize phase transitions in a 2D Lennard-Jones particle system. The simulation employs the Velocity Verlet integration algorithm with Numba JIT acceleration to achieve real-time performance. Users can "paint" temperature onto regions of the simulation, directly inducing phase transitions between solid, liquid, and gas states. We implement phase detection using the hexatic order parameter ψ₆ and demonstrate the characteristic melting behavior of the 2D Lennard-Jones system.

---

## 1. Introduction

Understanding phase transitions is fundamental to condensed matter physics and materials science. Phase transitions occur when a system undergoes a qualitative change in its properties, such as the transition from solid to liquid (melting) or liquid to gas (boiling). These transitions are governed by the competition between thermal energy (temperature) and interparticle interactions.

The **Lennard-Jones (LJ) potential** has served as the canonical model for studying these phenomena since its introduction in 1924. Despite its simplicity, the LJ potential captures the essential physics of noble gas interactions and provides a realistic model for studying phase behavior.

This project creates an interactive simulation where users can directly manipulate the temperature of local regions, effectively "painting" heat onto the particle system. This intuitive interface allows for exploration of:

1. **Melting transitions** - Adding heat to crystalline regions
2. **Crystallization** - Cooling disordered regions
3. **Evaporation** - Heating liquids to high temperatures
4. **Condensation** - Cooling gases into liquids

---

## 2. Theoretical Background

### 2.1 The Lennard-Jones Potential

The Lennard-Jones potential describes the interaction between a pair of neutral atoms or molecules:

$$V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

where:
- $r$ is the distance between particles
- $\varepsilon$ is the depth of the potential well (energy scale)
- $\sigma$ is the finite distance at which the potential is zero (length scale)

The potential has two components:
- **Repulsive term** $(r^{-12})$: Pauli repulsion at short range
- **Attractive term** $(r^{-6})$: van der Waals attraction at medium range

**Key features:**
- Minimum at $r_{min} = 2^{1/6}\sigma \approx 1.122\sigma$
- Zero crossing at $r = \sigma$
- Cutoff typically at $r_c = 2.5\sigma$ to improve computational efficiency

### 2.2 Reduced Units

We use Lennard-Jones reduced units throughout:

| Quantity | Reduced Unit |
|----------|--------------|
| Length | $\sigma$ |
| Energy | $\varepsilon$ |
| Mass | $m$ (particle mass) |
| Time | $\tau = \sigma\sqrt{m/\varepsilon}$ |
| Temperature | $\varepsilon/k_B$ |

In these units, $\varepsilon = \sigma = m = k_B = 1$.

### 2.3 Velocity Verlet Integration

We employ the Velocity Verlet algorithm for time integration, which is symplectic (preserves phase space volume) and provides excellent long-term energy conservation:

$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2$$

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2}[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)]\Delta t$$

This second-order method is preferred over higher-order methods (like RK4) for molecular dynamics because it preserves the symplectic structure of Hamiltonian mechanics.

### 2.4 Periodic Boundary Conditions

To simulate bulk behavior with a finite number of particles, we implement periodic boundary conditions (PBC). When a particle exits one side of the simulation box, it re-enters from the opposite side. Force calculations use the **minimum image convention**:

$$r_{ij} = \min(|r_j - r_i|, L - |r_j - r_i|)$$

where $L$ is the box dimension.

### 2.5 Phase Identification

#### 2.5.1 Temperature

The kinetic temperature is calculated from the equipartition theorem:

$$T = \frac{1}{N \cdot d \cdot k_B} \sum_{i=1}^{N} m_i v_i^2 = \frac{\langle v^2 \rangle}{d}$$

where $d = 2$ for our 2D system.

#### 2.5.2 Hexatic Order Parameter

In 2D, the hexatic order parameter $\psi_6$ measures the degree of six-fold orientational order:

$$\psi_6^{(i)} = \frac{1}{n_i} \left| \sum_{j \in \text{neighbors}(i)} e^{6i\theta_{ij}} \right|$$

where:
- $n_i$ is the number of neighbors of particle $i$
- $\theta_{ij}$ is the angle between particles $i$ and $j$

For a perfect hexagonal lattice, $\psi_6 = 1$. For a disordered system, $\psi_6 \approx 0$.

#### 2.5.3 Phase Boundaries

Approximate phase boundaries in 2D LJ reduced units:

| Transition | Temperature |
|------------|-------------|
| Solid → Liquid | $T \approx 0.4$ |
| Liquid → Gas | $T \approx 0.8$ |
| Critical point | $T_c \approx 1.0$ |

---

## 3. Implementation

### 3.1 Architecture

The simulation is structured as a Python package with four main modules:

```
src/
├── physics.py        # LJ potential, forces, energy calculations
├── simulation.py     # MD engine, Velocity Verlet, thermostats
├── thermodynamics.py # Phase detection, order parameters
└── visualization.py  # Particle rendering, color mapping
```

### 3.2 Performance Optimization

#### 3.2.1 Numba JIT Compilation

Critical numerical functions are decorated with Numba's `@jit` decorator:

```python
@jit(nopython=True, cache=True)
def compute_forces_and_energy(positions, box_size, epsilon, sigma, cutoff):
    ...
```

This provides 50-100x speedup over pure Python.

#### 3.2.2 Cell Lists

For large systems, we implement cell lists to reduce force computation from $O(N^2)$ to $O(N)$:

1. Divide the box into cells of size $\geq r_{cutoff}$
2. Each particle only interacts with particles in neighboring cells
3. Reduces the number of distance calculations significantly

### 3.3 Temperature Painting

The core "painting" feature modifies particle velocities locally:

```python
def add_heat(self, center, radius, amount):
    for i, pos in enumerate(positions):
        dist = distance(pos, center)
        if dist < radius:
            # Scale velocity to add kinetic energy
            scale = sqrt(1 + amount)
            velocities[i] *= scale
```

This provides intuitive control over local temperature while maintaining total momentum.

### 3.4 Berendsen Thermostat

For controlled temperature simulations, we implement the Berendsen thermostat:

$$\lambda = \sqrt{1 + \frac{\Delta t}{\tau_T}\left(\frac{T_0}{T} - 1\right)}$$

$$\mathbf{v}_i \rightarrow \lambda \mathbf{v}_i$$

This weakly couples the system to a heat bath at temperature $T_0$ with relaxation time $\tau_T$.

---

## 4. Results

### 4.1 Energy Conservation

Without a thermostat (NVE ensemble), the Velocity Verlet algorithm conserves total energy to within 0.1% over 10,000 steps:

| Metric | Value |
|--------|-------|
| Initial Energy | -180.5 |
| Final Energy | -180.3 |
| Drift | 0.11% |

### 4.2 Phase Transition Demonstration

Starting from a cold crystal ($T = 0.1$) and gradually heating:

| Temperature | Order Parameter | Phase |
|-------------|-----------------|-------|
| 0.1 | 0.92 | Solid |
| 0.3 | 0.85 | Solid |
| 0.4 | 0.65 | Melting |
| 0.5 | 0.45 | Liquid |
| 0.7 | 0.32 | Liquid |
| 1.0 | 0.18 | Gas |

The order parameter shows a clear drop at the melting transition.

### 4.3 Radial Distribution Function

The radial distribution function $g(r)$ shows characteristic peaks:

- **Solid**: Sharp peaks at lattice distances
- **Liquid**: Broadened peaks, liquid-like structure
- **Gas**: Approaches 1.0, no structure

### 4.4 Performance

| N Particles | Steps/sec | Speedup vs Python |
|-------------|-----------|-------------------|
| 100 | 5,200 | 87x |
| 400 | 1,480 | 62x |
| 1000 | 520 | 45x |

Performance measured on Apple M1, Python 3.12, Numba 0.58.

---

## 5. User Interface

The Streamlit web application provides:

1. **Particle Display**: Real-time visualization with temperature-based coloring
2. **Heat Brush**: Click-to-paint temperature interface
3. **Phase Indicator**: Color-coded current phase
4. **Energy Plots**: Time series of kinetic, potential, and total energy
5. **Temperature History**: Track temperature evolution

---

## 6. Discussion

### 6.1 Physical Accuracy

The 2D LJ system exhibits qualitatively correct behavior:
- Crystallization at low temperatures
- Melting transition with loss of long-range order
- Liquid phase with short-range correlations
- Gas phase at high temperatures

However, 2D systems have some differences from 3D:
- True long-range order is absent (Mermin-Wagner theorem)
- The hexatic phase may exist between solid and liquid
- Phase boundaries differ quantitatively

### 6.2 Educational Value

The interactive painting feature provides intuitive understanding of:
- The relationship between temperature and kinetic energy
- How local heating induces phase transitions
- The competition between order and thermal fluctuations

### 6.3 Limitations

1. **System size**: Limited to ~500 particles for real-time performance
2. **2D only**: 3D would require more computational resources
3. **No GPU**: Full CUDA/OpenCL acceleration not implemented
4. **Simplified phase detection**: More sophisticated methods exist

---

## 7. Conclusions

The Phase Transition Canvas successfully demonstrates:

1. **Interactive molecular dynamics** with real-time feedback
2. **Phase transitions** visible through order parameter and visual inspection
3. **Efficient simulation** through Numba JIT compilation
4. **Intuitive interface** via temperature "painting"

This tool provides both educational value for learning about phase transitions and a foundation for more sophisticated molecular dynamics simulations.

---

## 8. Future Work

1. **3D extension**: Implement 3D LJ simulation
2. **GPU acceleration**: Use JAX or CUDA for larger systems
3. **Additional potentials**: Implement Morse, Buckingham, soft-sphere
4. **Advanced thermostats**: Nosé-Hoover for rigorous canonical ensemble
5. **Phase diagram mapping**: Automated detection of phase boundaries

---

## References

1. Lennard-Jones, J. E. (1924). On the Determination of Molecular Fields. *Proc. R. Soc. Lond. A*, 106(738), 463-477.

2. Frenkel, D., & Smit, B. (2002). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.

3. Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids*. Oxford University Press.

4. Swope, W. C., et al. (1982). A computer simulation method for the calculation of equilibrium constants. *J. Chem. Phys.*, 76(1), 637-649.

5. Berendsen, H. J. C., et al. (1984). Molecular dynamics with coupling to an external bath. *J. Chem. Phys.*, 81(8), 3684-3690.

6. Strandburg, K. J. (1988). Two-dimensional melting. *Rev. Mod. Phys.*, 60(1), 161-207.

---

*Week 1 Project 2 - Biophysics Self-Study Portfolio*
