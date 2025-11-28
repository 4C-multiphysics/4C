#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install trilinos
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS:=4}
# git sha from Trilinos repository:
VERSION="f4d642715185dca1b94c91f434a2cf6db9f82014"
#CHECKSUM=""


# Location of script to apply patches later
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CMAKE_COMMAND=cmake

git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout $VERSION
cd .. && mkdir trilinos_build && cd trilinos_build

MPI_DIR=/usr
MPI_BIN_DIR=$MPI_DIR/bin

$CMAKE_COMMAND \
  -D CMAKE_BUILD_TYPE:STRING="RELEASE" \
  -D CMAKE_CXX_STANDARD:STRING="17" \
  -D CMAKE_CXX_COMPILER:FILEPATH="$MPI_BIN_DIR/mpic++" \
  -D CMAKE_C_COMPILER:FILEPATH="$MPI_BIN_DIR/mpicc" \
  -D CMAKE_Fortran_COMPILER:FILEPATH="$MPI_BIN_DIR/mpif90" \
  -D CMAKE_INSTALL_PREFIX:STRING=$INSTALL_DIR \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  \
  -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
  -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
  \
  -D Trilinos_ASSERT_MISSING_PACKAGES=OFF \
  -D Trilinos_ENABLE_Gtest:BOOL=OFF \
  -D Trilinos_ENABLE_Amesos:BOOL=ON \
    -D Amesos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Amesos2:BOOL=ON \
  -D Trilinos_ENABLE_AztecOO:BOOL=ON \
    -D AztecOO_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Belos:BOOL=ON \
  -D Trilinos_ENABLE_Epetra:BOOL=ON \
    -D Epetra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_EpetraExt:BOOL=ON \
    -D EpetraExt_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Intrepid2:BOOL=ON \
  -D Trilinos_ENABLE_Ifpack:BOOL=ON \
    -D Ifpack_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
  -D Trilinos_ENABLE_Isorropia:BOOL=ON \
    -D Isorropia_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Kokkos:BOOL=ON \
  -D Trilinos_ENABLE_ML:BOOL=ON \
    -D ML_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_MueLu:BOOL=ON \
  -D Trilinos_ENABLE_NOX:BOOL=ON \
  -D Trilinos_ENABLE_Sacado:BOOL=ON \
  -D Trilinos_ENABLE_SEACASExodus:BOOL=ON \
  -D Trilinos_ENABLE_SEACASNemesis:BOOL=OFF \
  -D Trilinos_ENABLE_Shards:BOOL=ON \
  -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
  -D Trilinos_ENABLE_Teko:BOOL=ON \
  -D Trilinos_ENABLE_Teuchos:BOOL=ON \
  -D Trilinos_ENABLE_Thyra:BOOL=ON \
    -D Thyra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON \
  -D Trilinos_ENABLE_ThyraEpetraExtAdapters:BOOL=ON \
  -D Trilinos_ENABLE_Tpetra:BOOL=ON \
    -D Tpetra_INST_INT_INT:BOOL=ON \
  -D Trilinos_ENABLE_Xpetra:BOOL=ON \
    -D Xpetra_ENABLE_Epetra:BOOL=ON \
    -D Xpetra_ENABLE_EpetraExt:BOOL=ON \
    -D Xpetra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
  -D Trilinos_ENABLE_Zoltan:BOOL=ON \
  -D Trilinos_ENABLE_Zoltan2:BOOL=ON \
  \
  -D Trilinos_MUST_FIND_ALL_TPL_LIBS=TRUE \
  -D TPL_ENABLE_DLlib:BOOL=OFF \
  -D TPL_ENABLE_Netcdf:BOOL=ON \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D TPL_ENABLE_ParMETIS:BOOL=ON \
    -D ParMETIS_INCLUDE_DIRS:PATH="/usr/include" \
  -D TPL_ENABLE_UMFPACK:BOOL=ON \
    -D UMFPACK_INCLUDE_DIRS:FILEPATH="/usr/include/suitesparse" \
  -D TPL_ENABLE_SuperLUDist:BOOL=ON \
    -D SuperLUDist_INCLUDE_DIRS:PATH="$INSTALL_DIR/include" \
    -D SuperLUDist_LIBRARY_DIRS:PATH="$INSTALL_DIR/lib" \
  \
  ../Trilinos

make -j${NPROCS} install
cd ..
rm -rf Trilinos trilinos_build
