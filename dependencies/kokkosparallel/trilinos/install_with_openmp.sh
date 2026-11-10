#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install trilinos with the OpenMP backend enabled
# Call with
# ./install_with_openmp.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS:=4}
# git sha from Trilinos repository (most recent supported version):
VERSION=${TRILINOS_VERSION:-"13519bfdf5504923a2b0160af7c83d837b5944a7"}
#CHECKSUM=""

# Location of this script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CMAKE_COMMAND=cmake

# Enable Trilinos debug assertions (can be overridden by environment)
TRILINOS_ENABLE_DEBUG=${TRILINOS_ENABLE_DEBUG:-ON}

source "$SCRIPT_DIR/../../common/trilinos_options.sh"

git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout $VERSION
cd .. && mkdir trilinos_build && cd trilinos_build

MPI_DIR=/usr
MPI_BIN_DIR=$MPI_DIR/bin

$CMAKE_COMMAND \
  -D CMAKE_CXX_COMPILER:FILEPATH="$MPI_BIN_DIR/mpic++" \
  -D CMAKE_C_COMPILER:FILEPATH="$MPI_BIN_DIR/mpicc" \
  -D CMAKE_Fortran_COMPILER:FILEPATH="$MPI_BIN_DIR/mpif90" \
  -D CMAKE_INSTALL_PREFIX:STRING=$INSTALL_DIR \
  "${TRILINOS_COMMON_OPTIONS[@]}" \
  \
  -D TPL_ENABLE_MUMPS:BOOL=ON \
  -D TPL_ENABLE_ScaLAPACK:BOOL=ON \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_ParMETIS:BOOL=ON \
    -D ParMETIS_INCLUDE_DIRS:PATH="/usr/include" \
  -D TPL_ENABLE_UMFPACK:BOOL=ON \
    -D UMFPACK_INCLUDE_DIRS:FILEPATH="/usr/include/suitesparse" \
  -D TPL_ENABLE_SuperLUDist:BOOL=ON \
    -D SuperLUDist_INCLUDE_DIRS:PATH="$INSTALL_DIR/../include" \
    -D SuperLUDist_LIBRARY_DIRS:PATH="$INSTALL_DIR/../lib" \
  \
  -D Trilinos_ENABLE_KokkosKernels:BOOL=ON \
  -D KokkosKernels_ENABLE_TPL_BLAS:BOOL=ON \
  -D KokkosKernels_ENABLE_TPL_LAPACK:BOOL=ON \
  \
  -D Kokkos_ENABLE_SERIAL:BOOL=ON \
  -D Tpetra_INST_SERIAL:BOOL=ON \
  \
  -D Trilinos_ENABLE_OpenMP:BOOL=ON \
  -D Kokkos_ENABLE_OPENMP:BOOL=ON \
  \
  ../Trilinos

make -j${NPROCS} install
cd ..
rm -rf Trilinos trilinos_build
