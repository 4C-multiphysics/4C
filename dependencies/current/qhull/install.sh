#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install qhull
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
VERSION="2012.1"
CHECKSUM="cb1296fbb9ec8b7d6e8f4c239ad165590616f242c7c46f790c27d8dcebe96c6a"

# Install qhull 2012.1
wget --no-verbose https://github.com/qhull/qhull/archive/refs/tags/${VERSION}.tar.gz
# Verify checksum
if [ $CHECKSUM = `sha256sum ${VERSION}.tar.gz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf ${VERSION}.tar.gz
cd qhull-${VERSION}/build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ..
make -j${NPROCS} && make install
cd ../../
rm -rf qhull* ${VERSION}.tar.gz