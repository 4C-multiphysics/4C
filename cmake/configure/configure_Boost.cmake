# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(
  Boost
  COMPONENTS graph system
  REQUIRED
  )

# post-process found targets
if(Boost_FOUND)
  message(STATUS "Boost component libraries: ${Boost_LIBRARIES}")
  message(STATUS "Boost libraries directory: ${Boost_LIBRARY_DIRS}")

  target_compile_definitions(
    Boost::graph
    INTERFACE "-DBOOST_MAJOR_VERSION=${Boost_MAJOR_VERSION}"
              "-DBOOST_MINOR_VERSION=${Boost_MINOR_VERSION}"
    )
  target_compile_definitions(
    Boost::system
    INTERFACE "-DBOOST_MAJOR_VERSION=${Boost_MAJOR_VERSION}"
              "-DBOOST_MINOR_VERSION=${Boost_MINOR_VERSION}"
    )

  target_link_libraries(
    four_c_all_enabled_external_dependencies INTERFACE Boost::system Boost::graph
    )

  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/templates/Boost.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/templates/Boost.cmake
    @ONLY
    )
endif()
