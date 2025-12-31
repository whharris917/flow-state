# 🌊 Flow State

**Flow State** (codenamed *Pipe Dream*) is a high-performance, Numba-accelerated 2D engineering simulator and CAD hybrid.

It bridges the gap between **Computer-Aided Design (CAD)** and **Computational Fluid Dynamics (CFD)**. Unlike traditional simulators that treat boundaries as static walls, Flow State implements **Bi-Directional Momentum Transfer**, allowing users to draft precise geometric structures and "atomize" them into reactive, physical rigid bodies that interact with thousands of fluid particles in real-time.

---

## 🚀 The Vision

In most particle engines, a wall is just a mathematical condition (`if x < 0: flip velocity`). In **Flow State**, a wall is a physical object with mass, inertia, and surface friction.

* **Draft:** Use precise CAD constraints (Parallel, Perpendicular, Tangent) to design complex mechanisms like turbines, pumps, or gears.
* **Atomize:** With one click, the geometry is discretized into a "skin" of particles tethered to a rigid "bone."
* **Simulate:** When a stream of high-velocity fluid strikes your structure, it doesn't just deflect—it absorbs energy. Turbines spin, gears mesh, and pistons compress.

---

## 🎮 Key Features

* **Hybrid Engine:** Seamlessly switches between a Geometric Constraint Solver (for editing) and a Verlet Particle Integrator (for simulation).
* **CAD Drafting Tools:** Professional-grade vector tools including Lines, Circles, Rectangles, and Parametric Constraints.
* **Dynamic Atomization:** Instantly convert static vector drawings into dynamic physical objects.
* **Live Inspection:** Tweak physics constants (Gravity, Viscosity, Temperature) in real-time while the simulation runs.
* **Numba Acceleration:** The core physics loop is JIT-compiled to machine code, enabling 60 FPS performance with thousands of particles on the CPU.

---

## 🛠 Technical Architecture

Flow State is built on a strict **"Two-Domain"** architecture to ensure stability and deterministic behavior.

### 1. The "Skin & Bone" Coupling
We use a proprietary **Spring-Tether System** to manage rigid-body dynamics:
* **The Bone (Geometry):** The source of truth for position/orientation, governed by a Rigid Body Solver.
* **The Skin (Atoms):** A collection of particles tethered to the geometry via high-stiffness springs.
* **The Result:** Smooth, continuous surfaces that interact physically with the fluid medium.

### 2. The "Air Gap" Principle
To guarantee undo/redo stability, the application enforces a strict separation between UI and Logic:
* **The UI** (Tools, Widgets) never mutates the model directly.
* **The Command Queue** is the only authorized gatekeeper for state changes.
* This ensures that every action is reversible and replayable.

### 3. Addition via Generalization (AvG)
The codebase adheres to the **AvG Principle**: we solve the general class of problems, not specific instances. We prioritize robust protocols (like `OverlayProvider` for UI) over ad-hoc fixes, keeping the codebase clean and modular.

---

## 🏁 Quick Start

### Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/FlowState.git](https://github.com/YourUsername/FlowState.git)

# Install dependencies (Numpy, Numba, Pygame)
pip install -r requirements.txt

# Run the simulation
python main.py

```

### Modes of Operation

* **Dashboard:** The main entry point.
* **Simulation Mode:** `python main.py --mode sim`
* **Builder Mode:** `python main.py --mode builder`

### Basic Workflow

1. **Draw:** Use the **Line** and **Circle** tools to create a mechanism.
2. **Constrain:** Apply **Fix**, **Parallel**, or **Coincident** constraints to define motion rules.
3. **Atomize:** Select your geometry and click **"Make Dynamic"**.
4. **Spray:** Use the **Brush Tool** to inject fluid and watch your machine come to life.

---

## 🗺 Roadmap

* **Phase 4 (Current):** **Active Power** — Implementing Motorized Revolute Joints for driven turbines and gears.
* **Phase 5:** **Advanced Constraints** — Adding Prismatic (sliding) joints and complex linkages.
* **Phase 6:** **Thermodynamics** — Introducing heat transfer, phase changes (liquid ↔ gas), and thermal expansion.

---

## 🤝 Contributing

Flow State is an open-source project for those who love physics, optimization, and clean architecture.

We welcome contributions! Please review `CLAUDE.md` before submitting PRs—it contains our architectural "Constitution," including the **AvG Principle** and **Air Gap** rules.

---

*Engineered with 💙 and Numba.*