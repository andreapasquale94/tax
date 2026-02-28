#!/usr/bin/env bash
set -euo pipefail

EIGEN_VERSION="${1:-5.0.0}"
RUN_TMP="${RUNNER_TEMP:-$(mktemp -d)}"
EIGEN_SRC="${RUN_TMP}/eigen-${EIGEN_VERSION}"
EIGEN_INSTALL="${RUN_TMP}/eigen-install-${EIGEN_VERSION}"
EIGEN_ARCHIVE="${RUN_TMP}/eigen-${EIGEN_VERSION}.tar.gz"

echo "Installing Eigen ${EIGEN_VERSION}"
curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" -o "${EIGEN_ARCHIVE}"
mkdir -p "${EIGEN_SRC}"
tar -xzf "${EIGEN_ARCHIVE}" -C "${EIGEN_SRC}" --strip-components=1

cmake -S "${EIGEN_SRC}" -B "${EIGEN_SRC}/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${EIGEN_INSTALL}"
cmake --build "${EIGEN_SRC}/build" --target install -j 2

if [[ -n "${GITHUB_ENV:-}" ]]; then
  if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    echo "CMAKE_PREFIX_PATH=${EIGEN_INSTALL}:${CMAKE_PREFIX_PATH}" >> "${GITHUB_ENV}"
  else
    echo "CMAKE_PREFIX_PATH=${EIGEN_INSTALL}" >> "${GITHUB_ENV}"
  fi
  echo "Exported CMAKE_PREFIX_PATH into GITHUB_ENV"
else
  echo "Eigen installed at: ${EIGEN_INSTALL}"
  echo "Set CMAKE_PREFIX_PATH to use it:"
  echo "  export CMAKE_PREFIX_PATH=${EIGEN_INSTALL}:\${CMAKE_PREFIX_PATH:-}"
fi
