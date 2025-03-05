# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

message(STATUS "Fetch content for magic_enum")
fetchcontent_declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_TAG e046b69a3736d314fad813e159b1c192eaef92cd # version 0.9.7
  )
set(MAGIC_ENUM_OPT_INSTALL
    ON
    CACHE BOOL "Turn on magic_enum install" FORCE
    )
fetchcontent_makeavailable(magic_enum)
set(FOUR_C_MAGIC_ENUM_ROOT "${CMAKE_INSTALL_PREFIX}/share/cmake/magic_enum")

four_c_add_external_dependency(four_c_all_enabled_external_dependencies magic_enum::magic_enum)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/magic_enum.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/templates/magic_enum.cmake
  @ONLY
  )
