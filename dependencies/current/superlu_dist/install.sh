#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install superLU_dist
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
VERSION="7.2.0"
CHECKSUM="20b60bd8a3d88031c9ce6511ae9700b7a8dcf12e2fd704e74b1af762b3468b8c"

SUPERLU_TAR="superlu_dist-${VERSION}.tar.gz"

wget --no-verbose https://github.com/xiaoyeli/superlu_dist/archive/refs/tags/v${VERSION}.tar.gz --output-document "$SUPERLU_TAR"

# Verify checksum
if [ $CHECKSUM = `sha256sum "$SUPERLU_TAR" | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf $SUPERLU_TAR
SUPERLU_SRC="superlu_dist-${VERSION}"
BUILD_DIR="${SUPERLU_SRC}-build"
mkdir -p $BUILD_DIR && cd $BUILD_DIR

CMAKE_COMMAND=cmake
MPI_DIR=/usr
MPI_BIN_DIR=$MPI_DIR/bin

$CMAKE_COMMAND \
  -DCMAKE_C_COMPILER=$MPI_BIN_DIR/mpicc \
  -DCMAKE_C_FLAGS="-std=c99 -O3 -g -DPRNTlevel=0 -DDEBUGlevel=0" \
  -DCMAKE_CXX_COMPILER=$MPI_BIN_DIR/mpicxx \
  -DCMAKE_CXX_FLAGS="-std=c++17" \
  -DCMAKE_Fortran_COMPILER=$MPI_BIN_DIR/mpif90 \
  \
  -Denable_openmp=OFF \
  \
  -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
  -DTPL_ENABLE_LAPACKLIB=ON \
  \
  -DTPL_PARMETIS_INCLUDE_DIRS="/usr/include" \
  -DTPL_PARMETIS_LIBRARIES="/usr/lib/libparmetis.so.4.0.3;/usr/lib/x86_64-linux-gnu/libmetis.so.5.1.0" \
  \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  ../$SUPERLU_SRC

make -j${NPROCS} install
cd ..
rm -rf $SUPERLU_TAR $SUPERLU_SRC $BUILD_DIR
