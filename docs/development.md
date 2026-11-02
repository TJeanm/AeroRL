# Local development

Use native arm64 Python 3.12 on Apple Silicon. The repository's `.venv` is isolated
from the system/Conda environment. All dependencies are declared in `pyproject.toml`;
there is no separate requirements file.

From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
CFLAGS="-Dfdopen=fdopen" .venv/bin/python -m pip install -e '.[dev]'
```

PyBullet 3.2.7 must build from source for this Python/macOS combination. The
[upstream installation instructions](https://github.com/learnsyslab/gym-pybullet-drones#installation)
prescribe the compiler flag above. Apple's Command Line Tools must be available
(`xcrun --find clang`); if missing, install them with `xcode-select --install` first.
The first build can take several minutes.

`gym-pybullet-drones` is installed as a dependency from the maintained
`learnsyslab/gym-pybullet-drones` repository at the commit recorded in
`pyproject.toml`. Its simulator, Crazyflie assets and controller stay in
`site-packages`; no upstream source is vendored here. The commit is pinned because
the upstream main branch can change its API and dependency requirements. Other
dependencies use compatible version ranges; see `pip list` for the local resolution.

## Checks

```sh
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest
```

Tests use PyBullet DIRECT and need no display or GPU. They check imports,
PyTorch/NumPy interoperability, and initialization, stepping and cleanup of a
Crazyflie simulation with the upstream PID controller. No policy is constructed.

## Visual smoke test

```sh
.venv/bin/python scripts/smoke_crazyflie.py
```

This opens PyBullet's GUI for five seconds. One packaged Crazyflie CF2X holds
position at 0.5 m using upstream `CtrlAviary`, `Physics.PYB` and `DSLPIDControl`.
It runs in real time, records no files, and disconnects on completion or Ctrl-C.
Use `--duration 15` for a longer look or `--headless` for a quick display-free run.
Run the GUI command from a logged-in macOS desktop session, on the main thread.

Verified on macOS 27.0.1, Apple M3 (arm64), Python 3.12.11: all three automated
tests passed, as did dependency, lint and formatting checks. Both a five-second
headless run and a fifteen-second GUI run completed at the expected hover position;
the GUI reported the Apple M3 OpenGL renderer and shut down its threads cleanly.
The upstream environment emits two Gymnasium warnings when its float64 space bounds
are cast to float32. These warnings do not prevent simulation and are left visible.

## Layout

- `aerorl/`: importable project package; `envs/`, `control/`, `tracks/` and `utils/`
  are placeholders for later work.
- `scripts/`: manually runnable experiments and diagnostics.
- `tests/`: small automated checks that run without a GUI.
- `results/`: generated outputs, ignored by Git except its placeholder.

No custom environments, controllers, dynamics, tracks, rewards or learning
algorithms are implemented in this setup.
