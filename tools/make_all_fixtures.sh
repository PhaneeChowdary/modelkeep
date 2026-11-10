#!/usr/bin/env bash
# Save the test models again with each old scikit-learn version. uv downloads each version one time.
# The list starts at 1.0, because scikit-learn 0.24 has no build for Apple Silicon.
set -u
cd "$(dirname "$0")/.."
while read -r python version numpy scipy; do
  echo "scikit-learn $version on Python $python"
  uv run -q --no-project --python "$python" --with "scikit-learn==$version" --with "$numpy" --with "$scipy" --with joblib \
    python tools/make_fixtures.py tests/fixtures 2>&1 | tail -1
done <<'VERSIONS'
3.9 1.0.2 numpy<1.24 scipy<1.11
3.10 1.1.3 numpy<2 scipy
3.10 1.2.2 numpy<2 scipy
3.11 1.3.2 numpy<2 scipy
3.11 1.4.2 numpy<2 scipy
3.12 1.5.2 numpy scipy
3.12 1.6.1 numpy scipy
3.13 1.7.2 numpy scipy
3.13 1.8.0 numpy scipy
VERSIONS
