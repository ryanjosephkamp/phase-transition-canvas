# 🔬 Phase Transition Canvas

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phase-transition-canvas.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Interactive 2D molecular dynamics simulation with temperature painting for visualizing phase transitions**

*Week 1 Project 2 - Biophysics Self-Study Portfolio*

## 📖 Overview

The Phase Transition Canvas is an interactive web application that simulates a 2D Lennard-Jones particle system. Users can "paint" temperature onto regions of the simulation and watch in real-time as matter transitions between solid, liquid, and gas phases.

### Key Features

- 🧊 **Real-time molecular dynamics** with Lennard-Jones potential
- 🖌️ **Interactive temperature painting** - add or remove heat locally
- 🔵🟣🔴 **Phase detection** - automatic identification of solid, liquid, and gas
- 📊 **Energy tracking** - monitor kinetic, potential, and total energy
- ⚡ **Numba-accelerated** - JIT compilation for high performance
- 🌐 **Web-based** - runs in your browser via Streamlit

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ryanjosephkamp/phase-transition-canvas.git
cd phase-transition-canvas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Online Demo

Visit the live demo at: **[phase-transition-canvas.streamlit.app](https://phase-transition-canvas.streamlit.app)**

## 🔬 The Physics

### Lennard-Jones Potential

Particles interact via the Lennard-Jones potential, which captures both:
- **Short-range repulsion** (Pauli exclusion)
- **Long-range attraction** (van der Waals forces)

$$V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

Where:
- $\varepsilon$ = depth of potential well (energy scale)
- $\sigma$ = finite distance at which potential is zero (length scale)
- $r$ = distance between particles

### Velocity Verlet Integration

We use the symplectic Velocity Verlet algorithm for time integration:

$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2$$

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2}[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)]\Delta t$$

This preserves phase space volume and provides excellent long-term energy conservation.

### Phase Transitions

The simulation demonstrates three phases:

| Phase | Temperature | Characteristics |
|-------|-------------|-----------------|
| 🔵 Solid | T < 0.4 | Ordered crystal structure, high ψ₆ |
| 🟣 Liquid | 0.4 < T < 0.8 | Disordered but dense, intermediate ψ₆ |
| 🔴 Gas | T > 0.8 | Dispersed particles, low density |

The **hexatic order parameter** ψ₆ measures crystalline order:
- ψ₆ = 1: Perfect hexagonal lattice
- ψ₆ ≈ 0: Completely disordered

## 🏗️ Project Structure

```
week_1_project_2/
├── app.py                 # Streamlit web application
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py        # Package initialization
│   ├── physics.py         # Lennard-Jones forces and potentials
│   ├── simulation.py      # MD simulation engine
│   ├── thermodynamics.py  # Phase detection and analysis
│   └── visualization.py   # Particle rendering
├── tests/
│   ├── test_physics.py    # Physics module tests
│   └── test_simulation.py # Simulation tests
└── docs/
    ├── scientific_report.md    # Detailed scientific report
    └── w1p2_phase_canvas_ieee.tex  # IEEE-format paper
```

## 💻 Command Line Interface

```bash
# Run equilibration test
python main.py --test

# Run phase transition demonstration
python main.py --demo

# Create animation
python main.py --animate

# Launch Streamlit app
python main.py --app

# With custom parameters
python main.py --test --particles 200 --steps 10000
```

## 🎮 How to Use

1. **Initialize**: Set number of particles and initial state (solid/liquid/gas)
2. **Run**: Click ▶️ to start the simulation
3. **Paint Heat**: 
   - Use sliders to position the brush
   - Click 🔥 to add heat or ❄️ to remove heat
4. **Observe**: Watch particles transition between phases
5. **Analyze**: View real-time temperature and energy plots

## 📊 Performance

| Particles | Steps/sec (Numba JIT) | Steps/sec (Pure Python) |
|-----------|----------------------|------------------------|
| 100       | ~5000                | ~50                    |
| 400       | ~1500                | ~15                    |
| 1000      | ~500                 | ~2                     |

*Tested on Apple M1, Python 3.12*

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Further Reading

- [Frenkel & Smit, "Understanding Molecular Simulation"](https://www.sciencedirect.com/book/9780122673511/understanding-molecular-simulation)
- [Allen & Tildesley, "Computer Simulation of Liquids"](https://global.oup.com/academic/product/computer-simulation-of-liquids-9780198803195)
- [Lennard-Jones Potential (Wikipedia)](https://en.wikipedia.org/wiki/Lennard-Jones_potential)

## 👤 Author

**Ryan Kamp**  
University of Cincinnati  
Department of Computer Science  
📧 kamprj@mail.uc.edu  
🔗 [GitHub](https://github.com/ryanjosephkamp)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Part of the 27-week Biophysics Self-Study Portfolio*
