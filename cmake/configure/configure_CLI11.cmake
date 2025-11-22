# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

message(STATUS "Fetch content for CLI11")
fetchcontent_declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG bfffd37e1f804ca4fae1caae106935791696b6a9 # version 2.6.1
  )

# Temporarily change project name to trick CLI11 into believing it is the main project
# This is required because CLI11 only sets up the install rules when it is the main project
set(_old ${CMAKE_PROJECT_NAME})
set(CMAKE_PROJECT_NAME "CLI11")

set(CLI11_INSTALL
    ON
    CACHE BOOL ""
    )
set(CLI11_BUILD_DOCS
    OFF
    CACHE BOOL ""
    )
set(CLI11_BUILD_TESTS
    OFF
    CACHE BOOL ""
    )
set(CLI11_BUILD_EXAMPLES
    OFF
    CACHE BOOL ""
    )

fetchcontent_makeavailable(CLI11)

set(CMAKE_PROJECT_NAME "${_old}")

set(FOUR_C_CLI11_ROOT "${CMAKE_INSTALL_PREFIX}")

four_c_add_external_dependency(four_c_all_enabled_external_dependencies CLI11::CLI11)
four_c_remember_variable_for_install(FOUR_C_CLI11_ROOT)
