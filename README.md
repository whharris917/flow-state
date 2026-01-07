# A Note from Wil

The majority of content in this repo was generated with the help of AI, particularly Claude Code (Opus 4.5) and Gemini 3 Pro. 

This note, however, is not. This is my attempt to "step out from behind the curtain" and explain what, exactly, I'm trying to do here.

This project started straighforwardly enough: I wanted to build a game. I have a B.S. in chemical engineering and have had a "pipe dream" of creating a chemical engineering simulation game for at least a decade. When reliable AI coding assistants became widely available, creating this game was one of the first things that came to mind. I didn't start using AI consistently until April 2025, at which point I realized that AI was far more capable than I had realized, and that I was late to the party. After some project naming fumbles, I began working on my game, Flow State. The name "Pipe Dream" now survives for historical reasons (i.e., I didn't think of Flow State until after I created the repo.) Gemini suggested I use it as my project's "code name." Works for me!

I'm sure I don't need to convince anyone reading this that AI has improved palpably over the last year. The jump from Gemini 2.5 to Gemini 3 and Claude Code with Opus 4.5 was shocking and exciting.

---

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