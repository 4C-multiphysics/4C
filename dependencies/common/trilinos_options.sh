#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Common, platform-independent CMake options for building Trilinos.
#
# Source this file after TRILINOS_ENABLE_DEBUG has been set and after all
# variant-specific variables (e.g. MPI_BIN_DIR, INSTALL_DIR) that are used by
# the variant's own CMake options. The array contains only options that do not
# depend on the platform or the selected Kokkos backend.

TRILINOS_COMMON_OPTIONS=(
  "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
  "-DCMAKE_CXX_STANDARD:STRING=17"
  "-DBUILD_SHARED_LIBS:BOOL=ON"

  "-DTrilinos_ENABLE_DEBUG:BOOL=${TRILINOS_ENABLE_DEBUG}"
  "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
  "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
  "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
  "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
  "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"

  "-DTrilinos_ASSERT_MISSING_PACKAGES=OFF"
  "-DTrilinos_ENABLE_Gtest:BOOL=OFF"
  "-DTrilinos_ENABLE_Amesos:BOOL=ON"
  "-DAmesos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
  "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
  "-DAztecOO_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_Belos:BOOL=ON"
  "-DTrilinos_ENABLE_Epetra:BOOL=ON"
  "-DEpetra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
  "-DEpetraExt_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
  "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
  "-DIfpack_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
  "-DTrilinos_ENABLE_Kokkos:BOOL=ON"
  "-DTrilinos_ENABLE_ML:BOOL=ON"
  "-DML_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_MueLu:BOOL=ON"
  "-DTrilinos_ENABLE_NOX:BOOL=ON"
  "-DNOX_ENABLE_ABSTRACT_IMPLEMENTATION_EPETRA:BOOL=OFF"
  "-DNOX_ENABLE_STRATIMIKOS_EPETRA_STACK:BOOL=OFF"
  "-DTrilinos_ENABLE_Sacado:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
  "-DTrilinos_ENABLE_SEACASNemesis:BOOL=OFF"
  "-DTrilinos_ENABLE_Shards:BOOL=ON"
  "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
  "-DTrilinos_ENABLE_Teko:BOOL=ON"
  "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
  "-DTrilinos_ENABLE_Thyra:BOOL=ON"
  "-DThyra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_ThyraEpetraExtAdapters:BOOL=ON"
  "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
  "-DTpetra_INST_INT_INT:BOOL=ON"
  "-DTrilinos_ENABLE_Xpetra:BOOL=ON"
  "-DXpetra_ENABLE_Epetra:BOOL=ON"
  "-DXpetra_ENABLE_EpetraExt:BOOL=ON"
  "-DXpetra_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
  "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
  "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"

  "-DTrilinos_MUST_FIND_ALL_TPL_LIBS=TRUE"
  "-DTPL_ENABLE_DLlib:BOOL=OFF"
  "-DTPL_ENABLE_Netcdf:BOOL=ON"
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DTPL_ENABLE_ParMETIS:BOOL=ON"
  "-DTPL_ENABLE_UMFPACK:BOOL=ON"
  "-DTPL_ENABLE_SuperLUDist:BOOL=ON"
)
