#!/bin/bash

set -e

VERSION="3.31.12"
INSTALL_DIR="/opt/cmake-${VERSION}"
INSTALLER="/tmp/cmake-${VERSION}-linux-x86_64.sh"
URL="https://github.com/Kitware/CMake/releases/download/v${VERSION}/cmake-${VERSION}-linux-x86_64.sh"

curl -L -o "$INSTALLER" "$URL"

chmod +x "$INSTALLER"

sudo mkdir -p "$INSTALL_DIR"

sudo "$INSTALLER" \
    --prefix="$INSTALL_DIR" \
    --skip-license

sudo ln -sf "$INSTALL_DIR/bin/cmake" /usr/local/bin/cmake
sudo ln -sf "$INSTALL_DIR/bin/ctest" /usr/local/bin/ctest
sudo ln -sf "$INSTALL_DIR/bin/cpack" /usr/local/bin/cpack

rm -f "$INSTALLER"

cmake --version
which cmake
