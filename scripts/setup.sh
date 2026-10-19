#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Autolife-Planning Setup ==="

# 1. Check / install pixi
if ! command -v pixi &>/dev/null; then
    echo ">> pixi not found – installing..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi
echo ">> pixi $(pixi --version)"

# 2. Initialize submodules (in case the repo was cloned without --recursive)
echo ">> Initializing git submodules..."
git submodule update --init --recursive

# 3. Install the pixi environment
echo ">> Installing pixi environment..."
pixi install

# 4. Build C++ third-party libraries
echo ">> Building cricket..."
pixi run cricket-build

echo ">> Building foam..."
pixi run foam-build

echo ""
echo "=== Setup complete ==="
echo "Run examples with:  pixi run python examples/planning/subgroup.py"
