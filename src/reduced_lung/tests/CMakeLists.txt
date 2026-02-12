# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_tests()

set(SUPPORT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/4C_reduced_lung_3_aw_2_tu_fields.json
    ${CMAKE_CURRENT_SOURCE_DIR}/4C_reduced_lung_3_aw_2_tu_topology.4C.yaml
    )
four_c_add_support_files_to_test(${AUTO_DEFINED_TEST_NAME} SUPPORT_FILES ${SUPPORT_FILES})
