# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_tests()

if(FOUR_C_WITH_GOOGLETEST)
  set(_reduced_lung_simd_test_target unittests_reduced_lung)
  set(_reduced_lung_simd_test_name unittests_reduced_lung.require_simd)
  set(_reduced_lung_simd_gtest_filter
      "ReducedLungTreeLinearSolverTests.ForcedBatchLargeRigidAirwaysMatchSparseSolver:ReducedLungTreeLinearSolverTests.ForcedBatchLargeKelvinVoigtAirwaysMatchSparseSolver:ReducedLungTreeLinearSolverTests.ReusedForcedBatchMixedTreeSolverMatchesSparseSolver"
      )
  separate_arguments(_reduced_lung_simd_mpiexec_args UNIX_COMMAND ${_mpiexec_all_args_for_testing})
  add_test(
    NAME ${_reduced_lung_simd_test_name}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${_reduced_lung_simd_mpiexec_args} -np 1
      $<TARGET_FILE:${_reduced_lung_simd_test_target}>
      --gtest_filter=${_reduced_lung_simd_gtest_filter}
      --gtest_output=xml:unittest_reports/${_reduced_lung_simd_test_name}_report.xml
    )
  set_tests_properties(
    ${_reduced_lung_simd_test_name}
    PROPERTIES TIMEOUT
               ${UNITTEST_TIMEOUT}
               LABELS
               minimal
               PROCESSORS
               1
               ENVIRONMENT
               "OMP_NUM_THREADS=1;FOUR_C_REDUCED_LUNG_REQUIRE_SIMD_TREE_SOLVER=1"
    )
endif()
