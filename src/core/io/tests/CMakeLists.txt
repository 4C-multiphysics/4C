# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_tests()

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/tmp")
if(FOUR_C_WITH_GOOGLETEST)
  target_compile_definitions(
    ${AUTO_DEFINED_TEST_NAME} PRIVATE FOUR_C_IO_TEST_TMP_DIR="${PROJECT_BINARY_DIR}/tmp"
    )
endif()

file(GLOB_RECURSE SUPPORT_FILES CONFIGURE_DEPENDS test_files/*)
four_c_add_support_files_to_test(${AUTO_DEFINED_TEST_NAME} SUPPORT_FILES ${SUPPORT_FILES})
