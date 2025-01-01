# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(Backtrace REQUIRED)

if(Backtrace_FOUND)
  message(STATUS "Backtrace include directory: ${Backtrace_INCLUDE_DIR}")
  message(STATUS "Backtrace library directory: ${Backtrace_LIBRARY}")

  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE Backtrace::Backtrace)

  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/templates/Backtrace.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/templates/Backtrace.cmake
    @ONLY
    )
  include(GNUInstallDirs)
  install(
    FILES ${CMAKE_SOURCE_DIR}/cmake/modules/FindBacktrace.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C/modules
    )
endif()
