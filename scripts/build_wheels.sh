#!/bin/bash
# Build manylinux_2_28 wheels using Docker.
# Usage: bash scripts/build_wheels.sh
#
# Produces wheels in dist/wheels/ compatible with glibc 2.28+
# (Ubuntu 20.04+, RHEL 8+, Debian 11+, etc.)
#
# Builds autolife-vamp version-specific wheels for Python 3.10, 3.11, 3.12.

set -euo pipefail

DOCKER_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/dist/wheels"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Building manylinux_2_28 wheels inside Docker..."
echo "    Project: $PROJECT_DIR"
echo "    Output:  $OUTPUT_DIR"

docker run --rm \
  -v "$PROJECT_DIR":/project:ro \
  -v "$OUTPUT_DIR":/output \
  "$DOCKER_IMAGE" \
  bash -c '
    set -euo pipefail

    # ------------------------------------------------------------------
    # Step 1: Install Eigen3 headers (only build-time dep for VAMP)
    # ------------------------------------------------------------------
    echo "==> Installing Eigen3 headers..."
    curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C /tmp
    cmake -S /tmp/eigen-3.4.0 -B /tmp/eigen-build \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF > /dev/null 2>&1
    cmake --install /tmp/eigen-build > /dev/null 2>&1
    echo "    Eigen3 installed to /usr/local"

    # ------------------------------------------------------------------
    # Step 2: Copy project to writable location
    # (project is mounted read-only to avoid root-owned artifacts on host)
    # ------------------------------------------------------------------
    echo "==> Copying project..."
    cp -r /project /build
    cd /build

    # Clean stale CMake caches from host builds (paths differ inside container)
    rm -rf third_party/vamp/build/
    rm -rf build/

    # ------------------------------------------------------------------
    # Step 3: Build autolife-vamp version-specific wheels
    # ------------------------------------------------------------------
    PYTHON_VERSIONS="cp312-cp312 cp311-cp311 cp310-cp310"

    for PYVER in $PYTHON_VERSIONS; do
      echo ""
      echo "==> Building autolife-vamp wheel for $PYVER..."
      export PATH=/opt/python/$PYVER/bin:$PATH
      pip install build scikit-build-core nanobind numpy cmake ninja -q
      SKBUILD_WHEEL_PY_API="" python -m build third_party/vamp --wheel --outdir /tmp/raw-wheels/
      rm -rf third_party/vamp/build/
    done

    # ------------------------------------------------------------------
    # Step 4: Repair wheels — bundle .so and tag as manylinux_2_28
    # ------------------------------------------------------------------
    echo ""
    echo "==> Repairing autolife-vamp wheels..."
    for whl in /tmp/raw-wheels/autolife_vamp-*.whl; do
      echo ""
      echo "    --- $(basename $whl) ---"
      auditwheel show "$whl" || true
      echo ""
      echo "    Repairing..."
      auditwheel repair "$whl" -w /output/ \
        --plat manylinux_2_28_x86_64 \
        --exclude libstdc++.so.6 \
        --exclude libgcc_s.so.1
    done

    # ------------------------------------------------------------------
    # Step 5: Build autolife-planning wheel
    # (No KDL in this container → pytracik is skipped → pure Python wheel)
    # ------------------------------------------------------------------
    echo ""
    echo "==> Building autolife-planning wheel..."
    export PATH=/opt/python/cp312-cp312/bin:$PATH
    pip install pybind11 setuptools -q
    python -m build --wheel --outdir /output/

    echo ""
    echo "==> All wheels:"
    ls -lh /output/*.whl
  '

echo ""
echo "==> Wheels are in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.whl
