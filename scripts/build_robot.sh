#!/usr/bin/env bash
# Build planning-ready robot description from v2.0 URDF.
# All project-specific paths live here; the Python script is a pure tool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Note: we intentionally do NOT pass --repair-meshes. CoACD runs downstream
# and outputs each collision piece as a convex hull, which is already manifold
# and safe for foam. Repairing (often convex-hulling) the full mesh upstream
# would corrupt the geometry CoACD needs to decompose.
#
# We also do not pass --distribute-to: the third_party submodules are kept as
# unmodified header-only dependencies; all project-local spherization artefacts
# stay under resources/robot/autolife.
python "$SCRIPT_DIR/build_robot_description.py" \
    --urdf "$ROOT/assets/autolife_description/urdf/robot_v2_2.urdf" \
    --mesh-dir "$ROOT/assets/autolife_description/meshes/robot_v2_2" \
    --output-dir "$ROOT/resources/robot/autolife" \
    "$@"
