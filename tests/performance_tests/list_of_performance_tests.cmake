# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Solid mechanics performance test
four_c_performance_test(
  TEST_FILE
  solid_hex8.4C.yaml
  MESH
  solid_hex8.vtu
  NP_MINIMAL
  2
  NP_FULL
  4
  REQUIRED_DEPENDENCIES
  VTK
  COPY_FILES
  ${PROJECT_SOURCE_DIR}/tests/input_files/xml/multigrid/elasticity_template.xml
  ${PROJECT_SOURCE_DIR}/tests/input_files/xml/linear_solver/iterative_gmres_template.xml
  )
four_c_performance_test(
  TEST_FILE
  solid_tet10.4C.yaml
  MESH
  solid_tet10.vtu
  NP_MINIMAL
  2
  NP_FULL
  4
  REQUIRED_DEPENDENCIES
  VTK
  COPY_FILES
  ${PROJECT_SOURCE_DIR}/tests/input_files/xml/multigrid/elasticity_template.xml
  ${PROJECT_SOURCE_DIR}/tests/input_files/xml/linear_solver/iterative_gmres_template.xml
  )

if(DEFINED FOUR_C_PERFORMANCE_TESTS_COLLECTION_FILE
   AND NOT "${FOUR_C_PERFORMANCE_TESTS_COLLECTION_FILE}" STREQUAL ""
   )
  # Note: We need to allow empty results since ctest might just run this collection "test" without any prior performance tests.
  four_c_collect_performance_test_results(
    TARGET_FILE ${FOUR_C_PERFORMANCE_TESTS_COLLECTION_FILE} ALLOW_EMPTY
    )
endif()
