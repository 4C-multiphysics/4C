#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install trilinos
# Call with
# ./install.sh /path/to/install/dir /path/to/dependency/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS:=4}
# git sha from Trilinos repository (most recent supported version):
VERSION=${TRILINOS_VERSION:-"13519bfdf5504923a2b0160af7c83d837b5944a7"}
#CHECKSUM=""


# Location of script to apply patches later
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CMAKE_COMMAND=cmake

# Enable Trilinos debug assertions (can be overridden by environment)
TRILINOS_ENABLE_DEBUG=${TRILINOS_ENABLE_DEBUG:-ON}

source "$SCRIPT_DIR/../../common/trilinos_options.sh"

git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout $VERSION
cd .. && mkdir trilinos_build && cd trilinos_build

BREW_DIR=/opt/homebrew
DEP_DIR=$2
MPI_BIN_DIR=$BREW_DIR/bin
GFORTRAN_LIB_DIR=$(gfortran -print-file-name=libgfortran.dylib | xargs dirname)

$CMAKE_COMMAND \
  -D CMAKE_CXX_COMPILER:FILEPATH="$MPI_BIN_DIR/mpic++" \
  -D CMAKE_CXX_FLAGS:STRING="-Wno-invalid-specialization" \
  -D CMAKE_C_COMPILER:FILEPATH="$MPI_BIN_DIR/mpicc" \
  -D CMAKE_Fortran_COMPILER:FILEPATH="$MPI_BIN_DIR/mpif90" \
  -D CMAKE_INSTALL_PREFIX:STRING=$INSTALL_DIR \
  "${TRILINOS_COMMON_OPTIONS[@]}" \
  \
  -D Zoltan2_ENABLE_ParMETIS:BOOL=OFF \
  -D TPL_ENABLE_MUMPS:BOOL=ON \
    -D MUMPS_INCLUDE_DIRS:FILEPATH="$DEP_DIR/mumps/include;$DEP_DIR/parmetis/include;" \
    -D MUMPS_LIBRARY_DIRS:FILEPATH="$DEP_DIR/mumps/lib;$DEP_DIR/parmetis/lib;" \
    -D MUMPS_LIBRARY_NAMES:STRING="dmumps;mumps_common;pord;mpi_usempif08;mpi_usempi_ignore_tkr;mpi_mpifh;metis;parmetis;scalapack;" \
  -D TPL_ENABLE_ScaLAPACK:BOOL=ON \
  -D ScaLAPACK_INCLUDE_DIRS:PATH="$BREW_DIR/include" \
    -D ScaLAPACK_LIBRARY_DIRS:PATH="$BREW_DIR/lib" \
    -D ScaLAPACK_LIBRARY_NAMES:STRING="scalapack" \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_ParMETIS:BOOL=ON \
    -D ParMETIS_INCLUDE_DIRS:PATH="$DEP_DIR/parmetis/include" \
    -D ParMETIS_LIBRARY_DIRS:PATH="$DEP_DIR/parmetis/lib" \
  -D TPL_ENABLE_UMFPACK:BOOL=ON \
    -D UMFPACK_INCLUDE_DIRS:FILEPATH="$BREW_DIR/include/suitesparse" \
    -D UMFPACK_LIBRARY_DIRS:FILEPATH="$BREW_DIR/lib" \
  -D TPL_ENABLE_SuperLUDist:BOOL=ON \
    -D SuperLUDist_INCLUDE_DIRS:PATH="$DEP_DIR/superlu_dist/include" \
    -D SuperLUDist_LIBRARY_DIRS:PATH="$DEP_DIR/superlu_dist/lib" \
  -D Trilinos_EXTRA_LINK_FLAGS:STRING="-L${GFORTRAN_LIB_DIR} -lgfortran" \
  \
  ../Trilinos

make -j${NPROCS} install
cd ..
rm -rf Trilinos trilinos_build
