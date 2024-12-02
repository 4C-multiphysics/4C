# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(FFTW REQUIRED)

if(FFTW_FOUND)
  message(STATUS "FFTW include directory: ${FFTW_INCLUDE_DIR}")
  message(STATUS "FFTW libraries: ${FFTW_LIBRARY}")

  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE fftw::fftw)

  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/templates/FFTW.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/templates/FFTW.cmake
    @ONLY
    )
  include(GNUInstallDirs)
  install(
    FILES ${CMAKE_SOURCE_DIR}/cmake/modules/FindFFTW.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/4C/modules
    )
endif()
