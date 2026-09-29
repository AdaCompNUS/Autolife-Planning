# Getting Started

## Prerequisites

- **Linux** (x86_64)
- **Python** 3.8–3.14

## Installation

Pre-built wheels are available for Python 3.8–3.14 on Linux x86_64. No local compilation required:

```bash
pip install autolife-planning
```

## Verify installation

```python
from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.planning import create_planner

planner = create_planner("autolife")
goal = planner.sample_valid()
result = planner.plan(HOME_JOINTS.copy(), goal)
print(f"Planning {'succeeded' if result.success else 'failed'}")
```

## Building Wheels from Source

Release wheels (CPython 3.12–3.14, manylinux x86_64) are built by
cibuildwheel in CI — see `.github/workflows/release.yml`.  To build a
wheel for the current pixi environment:

```bash
pixi run build-pkg     # output in dist/
```

## Development Setup

For contributing or rebuilding C++ dependencies from source, use [pixi](https://pixi.sh):

```bash
git clone --recursive https://github.com/AdaCompNUS/Autolife-Planning.git
cd Autolife-Planning
bash scripts/setup.sh
```

Or manually:

```bash
git clone --recursive https://github.com/AdaCompNUS/Autolife-Planning.git
cd Autolife-Planning
pixi install
pixi run cricket-build
pixi run foam-build
bash scripts/download_assets.sh
```
