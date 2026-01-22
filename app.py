#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Phase Transition Canvas - Interactive Streamlit Application
================================================================================

Project:        Week 1 Project 2: Phase Transition Canvas
Module:         app.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 22, 2026
Last Updated:   January 22, 2026

License:        MIT License
================================================================================

This is the main Streamlit application for the Phase Transition Canvas.
Users can:
- Initialize particle systems in solid/liquid/gas states
- "Paint" temperature onto regions with the mouse
- Watch phase transitions happen in real-time
- View energy and thermodynamic analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
from typing import Tuple, Optional, List
from PIL import Image
import io
import base64

# Try to import image coordinates for interactive clicking
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

# Import our modules
from src.physics import LennardJonesParameters, calculate_temperature
from src.simulation import MDSimulation, SimulationConfig, create_crystal_simulation
from src.thermodynamics import (
    Phase, identify_phase, calculate_global_order_parameter,
    PhaseTransitionTracker, SOLID_LIQUID_TEMP, LIQUID_GAS_TEMP
)
from src.visualization import (
    VisualizationConfig, render_particles_streamlit,
    TEMPERATURE_CMAP, get_phase_indicator_color
)


# Page configuration
st.set_page_config(
    page_title="Phase Transition Canvas",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
.phase-indicator {
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    margin: 5px 0;
}
.solid-phase {
    background-color: #3b82f6;
    color: white;
}
.liquid-phase {
    background-color: #8b5cf6;
    color: white;
}
.gas-phase {
    background-color: #ef4444;
    color: white;
}
.metric-box {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    margin: 5px 0;
}
.info-text {
    font-size: 14px;
    color: #94a3b8;
}
/* Smooth simulation display container */
.sim-container {
    width: 500px;
    height: 500px;
    background-color: #1a1a2e;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}
.sim-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'simulation' not in st.session_state:
        st.session_state.simulation = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0
    if 'temperature_history' not in st.session_state:
        st.session_state.temperature_history = []
    if 'energy_history' not in st.session_state:
        st.session_state.energy_history = {'time': [], 'kinetic': [], 'potential': [], 'total': []}
    if 'phase_tracker' not in st.session_state:
        st.session_state.phase_tracker = PhaseTransitionTracker()
    if 'vis_config' not in st.session_state:
        st.session_state.vis_config = VisualizationConfig()
    if 'brush_mode' not in st.session_state:
        st.session_state.brush_mode = 'heat'  # 'heat' or 'cool'
    if 'brush_radius' not in st.session_state:
        st.session_state.brush_radius = 3.0
    if 'brush_intensity' not in st.session_state:
        st.session_state.brush_intensity = 0.5


def create_simulation(
    n_particles: int,
    box_size: float,
    initial_state: str,
    initial_temp: float
) -> MDSimulation:
    """Create a new simulation with specified parameters."""
    config = SimulationConfig(
        box_size=(box_size, box_size),
        dt=0.002,
        lj_params=LennardJonesParameters(),
        use_thermostat=False
    )
    
    sim = MDSimulation(config)
    
    if initial_state == "Crystal (Solid)":
        # Initialize as crystal
        nx = int(np.sqrt(n_particles))
        ny = n_particles // nx
        
        spacing = 1.1  # Close to equilibrium distance
        sim.initialize_crystal(n_particles_x=nx, n_particles_y=ny, spacing=spacing, 
                               temperature=initial_temp)
    elif initial_state == "Random (Liquid)":
        sim.initialize_random(n_particles=n_particles, temperature=initial_temp)
    else:  # Gas
        sim.initialize_random(n_particles=n_particles, temperature=initial_temp * 2)
    
    return sim


def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("🔬 Phase Transition Canvas")
    
    st.sidebar.markdown("""
    ---
    ### About This Simulation
    
    This interactive simulation demonstrates **phase transitions** in a 
    2D Lennard-Jones particle system. The particles interact via the
    Lennard-Jones potential:
    
    $$V(r) = 4\\epsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12} - \\left(\\frac{\\sigma}{r}\\right)^6 \\right]$$
    
    **Paint temperature** onto regions to watch matter transition between:
    - 🔵 **Solid**: Low temperature, ordered structure
    - 🟣 **Liquid**: Intermediate temperature, disordered but dense
    - 🔴 **Gas**: High temperature, dispersed particles
    
    ---
    """)
    
    # Simulation Setup
    st.sidebar.subheader("⚙️ Simulation Setup")
    
    n_particles = st.sidebar.slider(
        "Number of Particles",
        min_value=25, max_value=400, value=100, step=25,
        help="More particles = more realistic but slower"
    )
    
    box_size = st.sidebar.slider(
        "Box Size",
        min_value=10.0, max_value=40.0, value=20.0, step=2.0,
        help="Size of the simulation box"
    )
    
    initial_state = st.sidebar.selectbox(
        "Initial State",
        ["Crystal (Solid)", "Random (Liquid)", "Hot Gas"],
        help="Starting configuration"
    )
    
    initial_temp = st.sidebar.slider(
        "Initial Temperature",
        min_value=0.05, max_value=2.0, value=0.2, step=0.05,
        help="Starting temperature (reduced units)"
    )
    
    if st.sidebar.button("🚀 Initialize Simulation", use_container_width=True):
        sim = create_simulation(n_particles, box_size, initial_state, initial_temp)
        st.session_state.simulation = sim
        st.session_state.step_count = 0
        st.session_state.temperature_history = []
        st.session_state.energy_history = {'time': [], 'kinetic': [], 'potential': [], 'total': []}
        st.session_state.phase_tracker = PhaseTransitionTracker()
        st.session_state.running = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Heat Brush Controls
    st.sidebar.subheader("🖌️ Temperature Brush")
    
    st.session_state.brush_mode = st.sidebar.radio(
        "Brush Mode",
        ["🔥 Heat", "❄️ Cool"],
        help="Choose whether to add or remove heat when clicking"
    )
    
    st.session_state.brush_radius = st.sidebar.slider(
        "Brush Radius",
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="Size of the temperature brush"
    )
    
    st.session_state.brush_intensity = st.sidebar.slider(
        "Brush Intensity",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="How much heat to add/remove"
    )
    
    st.sidebar.markdown("---")
    
    # Visualization Options
    st.sidebar.subheader("🎨 Visualization")
    
    color_mode = st.sidebar.selectbox(
        "Color By",
        ["temperature", "speed", "order"],
        help="What property determines particle color"
    )
    st.session_state.vis_config.color_by = color_mode
    
    show_velocities = st.sidebar.checkbox(
        "Show Velocity Arrows",
        value=False,
        help="Display velocity direction for each particle"
    )
    st.session_state.vis_config.show_velocities = show_velocities
    
    st.sidebar.markdown("---")
    
    # Author Info
    st.sidebar.markdown("""
    ### 👤 Author
    **Ryan Kamp**  
    University of Cincinnati  
    Department of Computer Science  
    📧 kamprj@mail.uc.edu  
    🔗 [GitHub](https://github.com/ryanjosephkamp)
    
    ---
    *Week 1 Project 2: Biophysics Self-Study*
    """)


def get_click_position(box_size: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """
    Get click position from user interaction.
    
    Note: Streamlit doesn't support direct canvas click detection,
    so we use sliders as a workaround for the web version.
    """
    # This is a simplified approach - real implementation would use
    # streamlit-drawable-canvas or custom component
    return None


def run_simulation_step(n_steps: int = 10):
    """Run simulation for n steps and update history."""
    sim = st.session_state.simulation
    if sim is None:
        return
    
    for _ in range(n_steps):
        sim.step()
        st.session_state.step_count += 1
    
    # Update histories
    state = sim.state
    temp = calculate_temperature(state.velocities)
    
    st.session_state.temperature_history.append(temp)
    st.session_state.energy_history['time'].append(st.session_state.step_count * sim.config.dt)
    st.session_state.energy_history['kinetic'].append(state.kinetic_energy)
    st.session_state.energy_history['potential'].append(state.potential_energy)
    st.session_state.energy_history['total'].append(state.total_energy)
    
    # Keep history limited
    max_history = 500
    if len(st.session_state.temperature_history) > max_history:
        st.session_state.temperature_history = st.session_state.temperature_history[-max_history:]
        for key in st.session_state.energy_history:
            st.session_state.energy_history[key] = st.session_state.energy_history[key][-max_history:]


def render_phase_indicator(phase: Phase):
    """Render a colored phase indicator."""
    phase_classes = {
        Phase.SOLID: "solid-phase",
        Phase.LIQUID: "liquid-phase",
        Phase.GAS: "gas-phase",
        Phase.UNKNOWN: ""
    }
    
    phase_names = {
        Phase.SOLID: "🔵 SOLID",
        Phase.LIQUID: "🟣 LIQUID",
        Phase.GAS: "🔴 GAS",
        Phase.UNKNOWN: "❓ UNKNOWN"
    }
    
    css_class = phase_classes.get(phase, "")
    name = phase_names.get(phase, "UNKNOWN")
    
    st.markdown(f"""
    <div class="phase-indicator {css_class}">
        {name}
    </div>
    """, unsafe_allow_html=True)


def render_main_content():
    """Render the main simulation content."""
    sim = st.session_state.simulation
    
    if sim is None:
        # Welcome screen
        st.title("🔬 Phase Transition Canvas")
        st.markdown("""
        ## Welcome to the Phase Transition Canvas!
        
        This interactive simulation lets you explore **phase transitions** in a 2D 
        molecular system. You can "paint" temperature onto particles and watch as 
        the system transitions between solid, liquid, and gas phases.
        
        ### 🎯 Key Features:
        - **Real-time molecular dynamics** with Lennard-Jones potential
        - **Interactive temperature painting** - add or remove heat locally
        - **Phase detection** - see the current phase based on order and temperature
        - **Energy tracking** - monitor kinetic, potential, and total energy
        
        ### 🚀 Getting Started:
        1. Use the sidebar to configure your simulation
        2. Click **Initialize Simulation** to start
        3. Use the **Heat Brush** to paint temperature
        4. Watch phase transitions happen!
        
        ### 🔬 The Physics:
        Particles interact via the **Lennard-Jones potential**:
        - Repulsive at short range (prevents overlap)
        - Attractive at medium range (holds matter together)
        - Zero at long range (particles are independent)
        
        Phase transitions occur when:
        - 🔵 **Solid → Liquid**: Enough heat overcomes lattice binding
        - 🟣 **Liquid → Gas**: Particles gain escape velocity
        - ❄️ **Gas → Liquid → Solid**: Cooling allows condensation/crystallization
        
        ---
        *👈 Use the sidebar to begin!*
        """)
        return
    
    # Main simulation view
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Particle System")
        
        # Control buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            run_label = "▶️ Run" if not st.session_state.running else "⏸️ Pause"
            if st.button(run_label, use_container_width=True, key="run_pause_btn"):
                st.session_state.running = not st.session_state.running
                st.rerun()  # Immediately update button state
        
        with btn_col2:
            if st.button("⏭️ Step (x10)", use_container_width=True):
                run_simulation_step(10)
        
        with btn_col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.simulation = None
                st.session_state.running = False
                st.rerun()
        
        with btn_col4:
            st.metric("Steps", st.session_state.step_count)
        
        # Run simulation if running
        if st.session_state.running:
            run_simulation_step(5)
        
        # Create background image of particles
        state = sim.state
        img_bytes = render_particles_streamlit(
            state.positions,
            state.velocities,
            sim.config.box_size,
            st.session_state.vis_config
        )
        
        # Canvas size for display
        canvas_width = 500
        canvas_height = 500
        
        if CLICK_AVAILABLE:
            # Use image coordinates for interactive clicking
            brush_mode = st.session_state.get('brush_mode', '🔥 Heat')
            mode_text = "ADD HEAT 🔥" if "Heat" in brush_mode else "COOL ❄️"
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(img_bytes))
            pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            if st.session_state.running:
                # When running, use HTML img for smooth updates (no flashing)
                st.markdown(f"**Simulation running...** (pause to click and paint temperature)")
                
                # Convert to base64 for HTML embedding
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Display using HTML for smoother updates
                st.markdown(
                    f'<div class="sim-container"><img src="data:image/png;base64,{img_base64}"></div>',
                    unsafe_allow_html=True
                )
            else:
                # When paused, use clickable image component
                st.markdown(f"**🖌️ Click on simulation to {mode_text}** (change mode in sidebar)")
                
                # Use static key to prevent widget recreation
                coords = streamlit_image_coordinates(pil_image, key="sim_canvas")
                
                # Process click
                if coords is not None:
                    # Store click with timestamp to detect new clicks
                    click_id = f"{coords['x']}_{coords['y']}"
                    last_click_id = st.session_state.get('last_click_id', None)
                    
                    if last_click_id != click_id:
                        st.session_state.last_click_id = click_id
                        
                        canvas_x = coords["x"]
                        canvas_y = coords["y"]
                        
                        # Scale to simulation box
                        sim_x = (canvas_x / canvas_width) * sim.config.box_size[0]
                        sim_y = (1 - canvas_y / canvas_height) * sim.config.box_size[1]  # Flip Y
                        
                        # Apply heat or cooling
                        if "Heat" in brush_mode:
                            sim.add_heat(
                                center=(sim_x, sim_y),
                                radius=st.session_state.brush_radius,
                                amount=st.session_state.brush_intensity * 0.5
                            )
                        else:
                            cooling_factor = max(0.3, 1.0 - st.session_state.brush_intensity * 0.5)
                            sim.remove_heat(
                                center=(sim_x, sim_y),
                                radius=st.session_state.brush_radius,
                                factor=cooling_factor
                            )
                        st.rerun()  # Update display after click
        else:
            # Fallback: display image with slider controls
            st.image(img_bytes, use_container_width=True)
            
            # Temperature painting controls (fallback)
            st.markdown("### 🖌️ Paint Temperature")
            st.markdown("*Install `streamlit-image-coordinates` for click painting, or use sliders:*")
            
            paint_col1, paint_col2 = st.columns(2)
            
            with paint_col1:
                x_coord = st.slider("X Position", 0.0, float(sim.config.box_size[0]), 
                                   float(sim.config.box_size[0])/2, step=0.5)
            with paint_col2:
                y_coord = st.slider("Y Position", 0.0, float(sim.config.box_size[1]), 
                                   float(sim.config.box_size[1])/2, step=0.5)
            
            paint_btn1, paint_btn2 = st.columns(2)
            
            with paint_btn1:
                if st.button("🔥 Add Heat Here", use_container_width=True):
                    sim.add_heat(
                        center=(x_coord, y_coord),
                        radius=st.session_state.brush_radius,
                        amount=st.session_state.brush_intensity
                    )
                    st.rerun()
            
            with paint_btn2:
                if st.button("❄️ Remove Heat Here", use_container_width=True):
                    cooling_factor = max(0.1, 1.0 - st.session_state.brush_intensity)
                    sim.remove_heat(
                        center=(x_coord, y_coord),
                        radius=st.session_state.brush_radius,
                        factor=cooling_factor
                    )
                    st.rerun()
    
    with col2:
        st.subheader("Analysis")
        
        # Calculate current properties
        state = sim.state
        temp = calculate_temperature(state.velocities)
        n_particles = len(state.positions)
        density = n_particles / (sim.config.box_size[0] * sim.config.box_size[1])
        
        # Order parameter (simplified - use subset for speed)
        order_param = 0.5  # Default
        if n_particles <= 200:
            from src.thermodynamics import calculate_global_order_parameter
            order_param = calculate_global_order_parameter(
                state.positions, sim.config.box_size
            )
        
        # Identify phase
        phase_info = identify_phase(temp, density, order_param)
        
        # Phase indicator
        st.markdown("### Current Phase")
        render_phase_indicator(phase_info.phase)
        
        # Metrics
        st.markdown("### Thermodynamics")
        
        met1, met2 = st.columns(2)
        with met1:
            st.metric("Temperature", f"{temp:.3f}")
        with met2:
            st.metric("Order (ψ₆)", f"{order_param:.3f}")
        
        met3, met4 = st.columns(2)
        with met3:
            st.metric("KE", f"{state.kinetic_energy:.2f}")
        with met4:
            st.metric("PE", f"{state.potential_energy:.2f}")
        
        st.metric("Total Energy", f"{state.total_energy:.2f}")
        st.metric("Density", f"{density:.3f}")
        
        # Temperature history plot
        if len(st.session_state.temperature_history) > 1:
            st.markdown("### Temperature History")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(st.session_state.temperature_history, 'r-', linewidth=1)
            ax.axhline(y=SOLID_LIQUID_TEMP, color='blue', linestyle='--', 
                      alpha=0.5, label='Melting')
            ax.axhline(y=LIQUID_GAS_TEMP, color='red', linestyle='--', 
                      alpha=0.5, label='Boiling')
            ax.set_xlabel('Step')
            ax.set_ylabel('T')
            ax.legend(fontsize=8)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Energy history plot
        if len(st.session_state.energy_history['time']) > 1:
            st.markdown("### Energy History")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(st.session_state.energy_history['kinetic'], 'r-', 
                   label='KE', linewidth=1)
            ax.plot(st.session_state.energy_history['potential'], 'b-', 
                   label='PE', linewidth=1)
            ax.plot(st.session_state.energy_history['total'], 'w-', 
                   label='Total', linewidth=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy')
            ax.legend(fontsize=8)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Auto-refresh when running
    if st.session_state.running:
        time.sleep(0.05)  # Small delay
        st.rerun()


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
