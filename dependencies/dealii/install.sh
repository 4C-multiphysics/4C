#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install deal.II
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS:=4}
VERSION="6890ca740b6e51530ebb7ae7e4932977010ef2be" # version 9.6.2
#CHECKSUM=""


CMAKE_COMMAND=cmake

git clone https://github.com/dealii/dealii.git
cd dealii
git checkout $VERSION
mkdir build
cd build

# Install p4est as part of deal.II
wget --no-verbose https://p4est.github.io/release/p4est-2.8.tar.gz
../doc/external-libs/p4est-setup.sh p4est-2.8.tar.gz $INSTALL_DIR

MPI_DIR=/usr
MPI_BIN_DIR=$MPI_DIR/bin

$CMAKE_COMMAND \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_STANDARD:STRING="20" \
  -D CMAKE_CXX_COMPILER:FILEPATH="$MPI_BIN_DIR/mpic++" \
  -D CMAKE_C_COMPILER:FILEPATH="$MPI_BIN_DIR/mpicc" \
  -D CMAKE_INSTALL_PREFIX:STRING=$INSTALL_DIR \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D DEAL_II_COMPONENT_EXAMPLES=OFF \
  -D DEAL_II_WITH_HDF5=ON \
  -D DEAL_II_WITH_MPI=ON \
  -D DEAL_II_WITH_P4EST=ON \
  -D DEAL_II_WITH_TRILINOS=ON \
  -D DEAL_II_WITH_BOOST=ON \
  -D DEAL_II_FORCE_BUNDLED_BOOST=ON \
  -D TRILINOS_DIR=$INSTALL_DIR \
  -D P4EST_DIR=$INSTALL_DIR/FAST \
  ..

make -j${NPROCS} install
cd ../..
rm -rf dealii
