This `README.md` is designed to serve as the definitive manual for **Flow State**. It balances the high-level vision with the "under-the-hood" physics logic we perfected today, ensuring any developer or engineer who finds the repo understands the sophistication of the engine.

---

# 🌊 Flow State

**Flow State** is a high-performance, Numba-accelerated 2D engineering simulator and CAD hybrid. It allows users to draft precise geometric structures and "atomize" them into reactive, physical rigid bodies that interact with thousands of fluid particles in real-time.

## 🚀 The Vision

Most particle simulators treat boundaries as static, immovable walls. **Flow State** changes the paradigm by implementing **Bi-Directional Momentum Transfer**. When a stream of high-velocity particles strikes a CAD-drawn wheel, the wheel doesn't just block the flow—it absorbs the energy, develops torque, and begins to spin.

By bridging the gap between professional CAD drafting and molecular-dynamics-style physics, Flow State provides a unique sandbox for testing turbines, filtration systems, and mechanical pumps.

---

## 🛠 Technical Architecture

### 1. The "Skin & Bone" Physics Model

Flow State uses a proprietary **Spring-Tether System** to manage rigid-body dynamics.

* **The Bone (Geometry):** The source of truth for the entity’s position and orientation, governed by a custom Rigid Body Solver.
* **The Skin (Atoms):** A collection of particles () tethered to the geometry via high-stiffness springs ().
* **Intra-Entity Force Exclusion:** To ensure structural rigidity, atoms belonging to the same entity ignore one another's Lennard-Jones forces, preventing "atomic staggering" and ensuring a smooth CAD surface.

### 2. Two-Way Coupling & Stability

To maintain stability in high-energy scenarios, the engine employs several stabilization strategies:

* **Mass/Inertia Ratios:** The "Chassis" (CAD geometry) is automatically scaled to be **5x heavier** than its constituent atoms to prevent the "tail wagging the dog" phenomenon.
* **Temporal Resolution:** The simulation runs at **20 sub-steps per frame**, allowing the math to resolve high-stiffness tethers without numerical explosions.
* **Aggressive Damping:** Linear and angular damping (0.90) are applied to bleed off integration errors, ensuring objects feel heavy and solid.

### 3. The Unified Material System

The engine features a "Grand Unification" of material properties. Users can define custom materials with the following parameters:
| Property | Description |
| :--- | :--- |
| **Sigma ()** | The effective radius/size of the particles. |
| **Epsilon ()** | The depth of the potential well (attraction/viscosity). |
| **Mass ()** | The weight of each particle, determining momentum transfer. |
| **Tether ** | The stiffness of the bond between an atom and its parent geometry. |

---

## 🎮 Key Features

* **CAD Drafting:** Precision tools for drawing lines and circles with snapping and constraint support.
* **Dynamic Atomization:** Instantly convert any static drawing into a physical object with a single click.
* **Particle Brush:** Spray fluid into the scene with adjustable flow rates and material presets.
* **Live Property Inspector:** Tweak physics constants—like gravity, viscosity, or mass—while the simulation is running for immediate feedback.
* **Numba Acceleration:** Leveraging JIT-compiled Python kernels to run thousands of particles at 60 FPS on the CPU.

---

## 🏁 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/FlowState.git

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py

```

### Basic Workflow

1. **Draw:** Use the Circle or Line tool to create a structure.
2. **Atomize:** Right-click your geometry and select **"Atomize"**.
3. **Go Live:** Click **"Make Dynamic"** to release the structure into the physics world.
4. **Interact:** Use the **Brush Tool** to spray fluid and observe the mechanical response.

---

## 🗺 Roadmap

* **Phase 4 (Current):** Active Power — Implementing Motorized Revolute Joints for turbines and gears.
* **Phase 5:** Advanced Constraints — Adding Prismatic (sliding) joints and complex linkages.
* **Phase 6:** Thermal Dynamics — Introducing heat transfer and phase changes (liquid to gas).

---

## 🤝 Contributing

Flow State is an open-source project. If you are interested in computational physics, Numba optimization, or CAD UI design, we welcome your contributions!

---