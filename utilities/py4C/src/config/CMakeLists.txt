# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

add_library(py4C_config INTERFACE)
target_include_directories(py4C_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# Add the special config target as a dependency
target_link_libraries(py4C_config INTERFACE config_deps)

set_target_properties(py4C_config PROPERTIES POSITION_INDEPENDENT_CODE ON)
