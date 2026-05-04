# Flow State Test Suite

Regression net for the Flow State application. Started as part of CR-116 (free-form Flow State exploration) so we can hack on features during the beach trip without silently breaking what already works.

## Scope

Tests are headless. Pygame runs against the dummy SDL driver (set in `conftest.py` before any pygame import).

### Model + engine layer

| File | Subsystem | What's covered |
|------|-----------|----------------|
| `test_geometry.py` | `model/geometry.py` | Line / Circle / Point primitives, dynamics state, dict round-trips |
| `test_sketch.py` | `model/sketch.py` | Entity CRUD, constraint registration, conflict resolution, materials |
| `test_solver.py` | `model/solver.py` | Constraint satisfaction (LENGTH, RADIUS, COINCIDENT, HORIZONTAL, VERTICAL, PARALLEL, PERPENDICULAR, EQUAL, MIDPOINT) |
| `test_commands.py` | `core/commands.py`, `model/commands/*` | CommandQueue undo/redo/discard/supersede/merge; each command's execute+undo (incl. ResizeWorldCommand snapshot/restore for Ctrl+Z support) |
| `test_compiler.py` | `engine/compiler.py` | Atom emission for static/dynamic/reference/handle entities, COINCIDENT joint IDs, bounds-clip for out-of-bounds positions |
| `test_simulation.py` | `engine/simulation.py` | World resize, sync_entity_arrays, snapshot/restore, to_dict round-trip |
| `test_brush.py` | `engine/particle_brush.py` | Paint/erase semantics, tethered/static atom preservation |
| `test_scene.py` | `core/scene.py` | Orchestrator dirty flags, undo triggers rebuild, ProcessObject registration |
| `test_process_objects.py` | `model/process_objects.py` | Source handle lifecycle, spawn behavior, serialization |
| `test_persistence.py` | `core/scene.py` save/load | `.scn` and `.mdl` round-trip including ProcessObjects |

### UI layer

| File | Subsystem | What's covered |
|------|-----------|----------------|
| `test_camera.py` | `core/camera.py` | View transforms, zoom clamping, pan accumulation, stored views per mode, view-state dict round-trip |
| `test_selection.py` | `core/selection.py` | Entity / point selection, toggle semantics, `remap_after_deletion` index shifting, dict round-trip |
| `test_session.py` | `core/session.py` | Defaults, tool activate/deactivate, per-mode tool memory, `clear_interaction_state`, mid-drag tool switch, `focused_element` lifecycle |
| `test_utils_transforms.py` | `core/utils.py` | `screen_to_sim` ↔ `sim_to_screen` inverse, pan/zoom effects, `get_connected_group` (incl. transitive COINCIDENT chains), `is_group_anchored` |
| `test_constraint_builder.py` | `core/constraint_builder.py` | State machine: start / add_wall / add_point / reset / check_ready, multi/binary flags, `try_build_command` auto-trim |
| `test_tool_context.py` | `core/tool_context.py` | Air Gap facade integrity: reflective sweep for model-object leaks, underscore-only escape hatches, Servo facade four-key contract, `get_active_material` returns a copy, `iter_entities` yields read-only views |
| `test_widgets.py` | `ui/ui_widgets.py` | UIElement / UIContainer position propagation; Button click vs drag-off semantics, toggle, disabled (does not consume), hover; InputField cursor + typing + arrow keys + click-outside deactivation; ContextMenu option dispatch; ConfirmDialog (Enter-confirms / Enter-cancels-when-destructive / Escape always cancels / modal key absorption) |
| `test_tools.py` | `ui/tools.py`, `ui/source_tool.py` | Line / Rect / Circle / Point / Brush / Select tool state machines via FakeApp + ToolContext; SelectTool MOVE_WALL cancel preserves geometry (PROP-2025-001); LineTool click-click cancel; BrushTool right-click; SourceTool two-click workflow + ESC cancel |
| `test_input_dispatch.py` | dispatch invariants | Modal-stack absorption (key/mouse events absorbed by `ConfirmDialog` while open, including Ctrl+Z); `reset_interaction_state` recursion through nested UIContainers |

## Documented latent bugs (xfail, strict=True)

None currently — the three bugs surfaced through TU collaboration during
CR-116 EI-3 (`Simulation.restore()` count==0, `Simulation.step()` escape
filter, `Scene.load_scene()` topology dirty) were all fixed in the same
commit batch and the markers removed. New bugs surfaced in future
sessions should be pinned here as `@pytest.mark.xfail(strict=True)` so
they start failing as soon as the underlying fix lands.

## What's *not* covered

- **Renderer** (`ui/renderer.py`, `ui/process_object_renderer.py`) — pixel output is not a regression target.
- **App lifecycle** (`app/flow_state_app.py`, `app/app_controller.py`, `app/dashboard.py`) — full integration; covered by Lead-witnessed manual smoke during EI execution.
- **Input dispatch chain** (`ui/input_handler.py`) — would need most of the controller wired up. Modal stack push/pop and dialog absorption are tested at the dialog level (`test_widgets.py`).

Add coverage here as new seams emerge or when bugs are caught manually that should have a test.

## Running

```bash
# from flow-state/
pytest                           # full suite
pytest tests/test_compiler.py    # one file
pytest -k "undo"                 # by name pattern
pytest -m "not slow"             # skip Numba-warmup / real-physics tests
```

Most fixtures construct objects with `skip_warmup=True` to bypass Numba JIT (~5-10s on first run). Tests marked `@pytest.mark.slow` exercise the warmup path or run real physics steps.

## Conventions

- Tests are independent — no shared state between tests in the same file.
- Fixtures live in `conftest.py`. Prefer fresh `Scene` / `Sketch` per test.
- A test asserts a single behavior. If multiple assertions are needed to characterize one behavior, keep them in one test.
- Add a comment only when the *why* is non-obvious.
