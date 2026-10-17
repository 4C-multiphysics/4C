# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_process_global_option(
  FOUR_C_ARBORX_FIND_INSTALLED
  DESCRIPTION
  "Use installed ARBORX instead of fetching sources"
  DEFAULT
  OFF
  )
if(FOUR_C_ARBORX_FIND_INSTALLED)
  message(STATUS "FOUR_C_ARBORX_FIND_INSTALLED is enabled")

  # ARBORX provides a package configuration file if installed.
  find_package(ArborX HINTS ${FOUR_C_ARBORX_ROOT})

  if(NOT ArborX_FOUND)
    message(
      FATAL_ERROR
        "ArborX could not be found. Please ensure that the FOUR_C_ARBORX_ROOT path is correctly defined in the config file. Also, please use 'make install' and not just 'make' to install ArborX."
      )
  endif()
else() # Fetch ArborX from GIT repository
  message(STATUS "Fetch content for ArborX")
  # Unconditionally turn on MPI support inside ArborX
  set(ARBORX_ENABLE_MPI "ON")
  set(ARBORX_GIT_REPO "https://github.com/kokkos/ArborX.git")
  # Post v2.1, includes fix for empty BVH
  set(ARBORX_GIT_TAG "f804f078c9aefedfa9eed6127a5e6e9fb169f53b")

  fetchcontent_declare(
    arborx
    GIT_REPOSITORY ${ARBORX_GIT_REPO}
    GIT_TAG ${ARBORX_GIT_TAG}
    )
  fetchcontent_makeavailable(arborx)
  set(FOUR_C_ARBORX_ROOT "${CMAKE_INSTALL_PREFIX}")
endif()

four_c_add_external_dependency(four_c_all_enabled_external_dependencies ArborX::ArborX)

four_c_remember_variable_for_install(FOUR_C_ARBORX_FIND_INSTALLED FOUR_C_ARBORX_ROOT)
