# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

add_library(unittests_common INTERFACE)
target_include_directories(unittests_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
four_c_add_internal_dependency(unittests_common core)

add_subdirectory(tests)
