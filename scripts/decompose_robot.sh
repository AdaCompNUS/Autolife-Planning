#!/usr/bin/env bash
# Convex-decompose autolife collision meshes with CoACD so downstream foam
# spherization sees (approximately) convex input per piece.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Arm↔chest clearance is the motion-planner's sensitive zone: the upper-arm
# chain needs to move close to (and sometimes past) the torso without a huge
# conservative sphere on the chest blocking solutions. Decompose only these
# links — and only COARSELY: the goal is to split torso-body from shoulder-
# mount (and arm-body from joint housing) so foam never fits a single sphere
# spanning both. A handful of pieces per link is enough; finer decomposition
# wastes FK compute. Everything else keeps its raw mesh.
python -u "$SCRIPT_DIR/decompose_meshes.py" \
    --input   "$ROOT/resources/robot/autolife/autolife_base_simple.urdf" \
    --output  "$ROOT/resources/robot/autolife/autolife_base_decomposed.urdf" \
    --parts-dir "$ROOT/resources/robot/autolife/meshes/decomposed" \
    --threshold 0.2 \
    --max-convex-hull 6 \
    --include 'Link_Waist_Yaw_to_Shoulder_Inner' \
    --include 'Link_(Left|Right)_Shoulder_Inner_to_Shoulder_Outer' \
    --include 'Link_(Left|Right)_Shoulder_Outer_to_UpperArm' \
    --include 'Link_(Left|Right)_UpperArm_to_Elbow' \
    "$@"
