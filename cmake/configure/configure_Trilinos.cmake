# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Kokkos is typically pulled in via Trilinos. If no location has been given,
# try the same location as Trilinos. If no Trilinos location exists, users
# will get an error to provide that one first.
set(Kokkos_FIND_QUIETLY TRUE)
if(Trilinos_ROOT AND NOT Kokkos_ROOT)
  set(Kokkos_ROOT
      ${Trilinos_ROOT}
      CACHE PATH "Path to Kokkos installation"
      )
endif()

# We only support Trilinos versions that provide a config file.
find_package(Trilinos REQUIRED)

message(STATUS "Trilinos version: ${Trilinos_VERSION}")
message(STATUS "Trilinos packages: ${Trilinos_PACKAGE_LIST}")

if(Trilinos_FOUND AND NOT TARGET Trilinos::all_selected_libs)
  # In preparation for newer Trilinos releases, create a target
  # Trilinos::all_selected_libs with the correct dependencies
  add_library(Trilinos::all_selected_libs IMPORTED INTERFACE)
  target_include_directories(
    Trilinos::all_selected_libs
    SYSTEM
    INTERFACE ${Trilinos_INCLUDE_DIRS}
    INTERFACE ${Trilinos_TPL_INCLUDE_DIRS}
    )
  target_link_libraries(
    Trilinos::all_selected_libs INTERFACE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
    )
endif()

target_link_libraries(
  four_c_all_enabled_external_dependencies INTERFACE Trilinos::all_selected_libs
  )

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/Trilinos.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/templates/Trilinos.cmake
  @ONLY
  )
