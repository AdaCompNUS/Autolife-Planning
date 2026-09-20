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

To build distributable wheels for all supported Python versions:

```bash
bash scripts/build_wheels.sh
```

This uses Docker with the `manylinux_2_28` image to produce portable Linux wheels. The output goes to `dist/wheels/`. It builds:

- **autolife-vamp** — Version-specific wheels for Python 3.10, 3.11, 3.12
- **autolife-planning** — Pure Python wheel (works on any Python 3.10+)

Requirements: Docker must be installed and running.

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
