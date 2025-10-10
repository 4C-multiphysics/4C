# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(ZLIB REQUIRED)

# post-process found targets
if(ZLIB_FOUND)
  message(STATUS "ZLIB include directory: ${ZLIB_INCLUDE_DIRS}")
  message(STATUS "ZLIB libraries: ${ZLIB_LIBRARIES}")
  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE ZLIB::ZLIB)

  four_c_remember_variable_for_install(ZLIB_INCLUDE_DIRS ZLIB_LIBRARIES)
endif()
