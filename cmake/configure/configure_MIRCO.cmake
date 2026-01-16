# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_process_global_option(
  FOUR_C_MIRCO_FIND_INSTALLED
  DESCRIPTION
  "Use installed MIRCO instead of fetching sources"
  DEFAULT
  OFF
  )
if(FOUR_C_MIRCO_FIND_INSTALLED)

  message(STATUS "FOUR_C_MIRCO_FIND_INSTALLED is enabled")

  # MIRCO provides a package configuration file if installed.
  find_package(mirco_lib HINTS ${FOUR_C_MIRCO_ROOT})

  if(NOT mirco_lib_FOUND)
    message(
      FATAL_ERROR
        "mirco_lib could not be found. Please ensure that the FOUR_C_MIRCO_ROOT path is correctly defined in the config file. Also, please use 'make install' and not just 'make' to install MIRCO."
      )
  endif()

else() # Fetch MIRCO from GIT repository
  # Turn off googletest and Trilinos in MIRCO so that they don't interfere with 4C
  set(GTEST_IN_MIRCO "OFF")
  set(TRILINOS_IN_MIRCO "OFF")

  set(MIRCO_GIT_REPO "https://github.com/imcs-compsim/MIRCO.git")
  set(MIRCO_GIT_TAG "84d9fa94b3cd2d358f9c498e716d50181ba626f2")

  fetchcontent_declare(
    mirco
    GIT_REPOSITORY ${MIRCO_GIT_REPO}
    GIT_TAG ${MIRCO_GIT_TAG}
    )
  fetchcontent_makeavailable(mirco)
  # MIRCO requires a specific path, possibly due to inconsistent naming "mirco" vs "mirco_lib".
  set(FOUR_C_MIRCO_ROOT "${CMAKE_INSTALL_PREFIX}/lib/cmake/mirco")

  four_c_add_external_dependency(four_c_all_enabled_external_dependencies mirco::mirco_lib)
endif()

four_c_remember_variable_for_install(FOUR_C_MIRCO_ROOT)
