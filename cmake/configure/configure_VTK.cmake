# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML)

if(VTK_FOUND)
  message(STATUS "VTK component libraries: ${VTK_LIBRARIES}")

  target_link_libraries(
    four_c_all_enabled_external_dependencies
    INTERFACE VTK::CommonCore VTK::CommonDataModel VTK::IOXML
    )

  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/templates/VTK.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/templates/VTK.cmake
    @ONLY
    )
endif()
