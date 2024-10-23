# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

find_package(Qhull REQUIRED)

if(QHULL_FOUND)
  message(STATUS "QHULL include directory: ${QHULL_INCLUDE_DIR}")
  message(STATUS "QHULL library directory: ${QHULL_LIBRARY}")

  target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE qhull::qhull)
endif()
