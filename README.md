# 3D Fusion Plasma Particle-in-Cell (PIC) Simulation 

## 📋 Overview

This CUDA/C++ simulation implements a **fully-3D Particle-in-Cell (PIC) plasma physics code** specifically tailored for fusion energy research. The code simulates the complex behavior of hot, magnetically-confined plasmas using first-principles particle methods with GPU acceleration, providing real-time diagnostics and visualization of key plasma phenomena.

![Fusion Plasma Simulation](https://img.shields.io/badge/Physics-Fusion%20Plasma-blueviolet) ![CUDA Accelerated](https://img.shields.io/badge/Compute-CUDA%20Accelerated-green) ![OpenGL Visualization](https://img.shields.io/badge/Visualization-OpenGL-orange)


🎥 [CUDA Simulation](simulation_demos/fusion_demo.gif)

## 🎯 Key Features

### **Advanced Physics Capabilities**
- **Full Maxwell's Equations**: Self-consistent electromagnetic field evolution
- **Boris Particle Pusher**: Relativistically-correct particle motion
- **Coulomb Collisions**: Monte Carlo treatment with realistic cross-sections
- **Magnetic Confinement**: Toroidal geometry with Grad-Shafranov-like profiles
- **Debye Shielding**: Resolves plasma collective effects
- **Plasma Instabilities**: Captures drift-wave and MHD-type modes
- **Wave-Particle Interactions**: Self-consistent field-particle coupling

### **Comprehensive Diagnostics**
- **Energy Tracking**: Separated electron/ion kinetic, field energies
- **Temperature Profiles**: Core-edge temperature gradients
- **Density Fluctuations**: Quantitative turbulence measures
- **Plasma Parameters**: β, confinement times, collisionality
- **Phase Space Analysis**: Velocity and spatial distributions
- **Instability Growth Rates**: Real-time mode identification

### **Interactive Visualization**
- **12 Visualization Modes**: From particle trajectories to field structures
- **Real-time Control**: Interactive camera, rotation, and zoom
- **Diagnostic Overlay**: Heads-up display with key plasma parameters
- **GPU-accelerated Rendering**: Smooth visualization at scale

## 🧮 Physics Model

### Governing Equations
```
∂f/∂t + v·∇f + (q/m)(E + v×B)·∂f/∂v = C[f]  (Vlasov-Boltzmann)
∇·E = ρ/ε₀                                  (Poisson)
∂B/∂t = -∇×E                               (Faraday)
∂E/∂t = (∇×B)/μ₀ - J/ε₀                     (Ampère-Maxwell)
```

### Physical Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Core Temperature** | 15-20 keV | Fusion-relevant temperatures |
| **Density** | 5×10²⁰ m⁻³ | ITER-like density |
| **Magnetic Field** | 5 Tesla | Standard tokamak field |
| **Particle Count** | 3×10⁶ | Macro-particles (electrons + ions) |
| **Grid Resolution** | 10³ cells | 1 mm resolution |
| **Time Step** | 0.2 ps | Resolves electron plasma frequency |

## 🛠️ Installation & Building

### Prerequisites

**System Requirements:**
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.0 or higher
- OpenGL/GLUT libraries
- 8GB+ GPU memory recommended

**Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential nvidia-cuda-toolkit freeglut3-dev
```

### Building the Simulation

```bash
# Clone and compile
git clone https://github.com/yourusername/fusion-pic-simulation.git
cd fusion-pic-simulation

# Compile with CUDA
nvcc -std=c++14 -o fusion_pic plasma.cu -lglut -lGLU -lGL -lm -O3 -lcufft -arch=sm_60

# Run the simulation
./fusion_pic
```



### GPU Acceleration Strategy
- **Particle Operations**: 256 threads/block, distributed across all particles
- **Field Operations**: 8×8×8 thread blocks for 3D grid
- **Memory Layout**: Structure of Arrays (SoA) for coalesced access
- **Async Operations**: Overlap computation with PCIe transfers

## 🧪 Research Applications

### **Fusion Energy**
- Tokamak/Stellarator plasma modeling
- Fast ion behavior and alpha particle confinement
- Edge Localized Modes (ELMs) and disruptions
- Radio-frequency heating efficiency

### **Space Physics**
- Solar wind-magnetosphere interactions
- Magnetic reconnection studies
- Plasma turbulence in astrophysical jets

### **Industrial Applications**
- Plasma processing and etching
- Electric propulsion for spacecraft
- Plasma medicine and bio-applications


## 🐛 Known Issues & Limitations

### Current Limitations
- **Limited Grid Size**: 10³ cells due to memory constraints
- **Collision Operator**: Simplified Monte Carlo, not full Landau
- **Boundary Conditions**: Simplified perfect conductor boundaries
- **Non-relativistic**: v ≪ c assumed (valid for most fusion plasmas)

### Planned Improvements
- Adaptive mesh refinement (AMR)
- Full FDTD Maxwell solver
- Implicit time stepping for electrons
- MPI parallelization for multi-node runs
- Advanced collision operators (Fokker-Planck)


## 🎓 Educational Use

This code is suitable for:
- **Graduate-level plasma physics courses**
- **Computational physics research projects**
- **HPC training in scientific computing**
- **Visualization of complex physical systems**
