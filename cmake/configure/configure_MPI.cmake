# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# This find module is provided by CMake

# Disable deprecated CXX bindings in MPI. These often lead to compiler warnings.
set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED)

target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE MPI::MPI_CXX)

# Check if MPI works: Ubuntu 20.04 has a broken MPI installation. Try to patch it
# up by adding open-pal as an additional dependency.
four_c_check_compiles(FOUR_C_MPI_LINKAGE_OK LINK_LIBRARIES MPI::MPI_CXX)
if(NOT FOUR_C_MPI_LINKAGE_OK)
  four_c_check_compiles(FOUR_C_MPI_LINKAGE_OK_WITH_OPAL LINK_LIBRARIES MPI::MPI_CXX "open-pal")
  if(FOUR_C_MPI_LINKAGE_OK_WITH_OPAL)
    message(
      STATUS
        "MPI installation is underlinked. Adding open-pal as an additional dependency to fix this."
      )
    target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE "open-pal")
  else()
    message(FATAL_ERROR "Cannot build/link a program with MPI. Check your compiler settings.")
  endif()
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/MPI.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/templates/MPI.cmake
  @ONLY
  )
