# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

message(STATUS "Fetch content for Fastor")

fetchcontent_declare(
  Fastor
  GIT_REPOSITORY https://github.com/romeric/Fastor
  GIT_TAG 652972f981f51f503b4f66f7190d5bd69b980dee # some version
  )

set(FASTOR_INSTALL
    ON
    CACHE BOOL "Turn on FASTOR install" FORCE
    )

fetchcontent_makeavailable(Fastor)

set(FOUR_C_FASTOR_ROOT "${CMAKE_INSTALL_PREFIX}/lib/cmake/fastor")

four_c_add_external_dependency(four_c_all_enabled_external_dependencies Fastor)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/templates/Fastor.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/templates/Fastor.cmake
  @ONLY
  )
