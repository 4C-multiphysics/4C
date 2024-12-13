# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#
# This executable may be run in serial or with NP*THREADS processors given as arguments
#
# Usage: four_c_add_google_test_executable(
#   <name>
#   SOURCE source1 [source2 ...]
#   [NP <number of MPI-processes>]
#   [THREADS <number of OpenMP threads>]
#   [SUPPORT_FILES support_file1 [support_file2 ...]])
#
# Arguments:
#   <name> (required): The name of the test executable
#   SOURCE (required): The source files of the test executable
#   NP (optional): The number of MPI processes to use (default: 1)
#   THREADS (optional): The number of OpenMP threads to use (default: 1)
#   SUPPORT_FILES (optional): Additional files to be copied to the test directory. These files can be used as input
#     files inside the tests and should be referred to relative to the current CMakeLists.txt file.
#
# Note: This function will do nothing if unit tests are not configured.
function(four_c_add_google_test_executable TESTNAME)
  if(NOT FOUR_C_WITH_GOOGLETEST)
    return()
  endif()

  set(options "")
  set(oneValueArgs NP THREADS)
  set(multiValueArgs SOURCE SUPPORT_FILES)
  cmake_parse_arguments(
    FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  if(DEFINED FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_UNPARSED_ARGUMENTS)
    message(
      SEND_ERROR
        "There are unparsed arguments: ${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_UNPARSED_ARGUMENTS}"
      )
  endif()

  if(NOT DEFINED FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_SOURCE)
    message(SEND_ERROR "Need to specify at least one source file.")
  endif()

  if(NOT DEFINED FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP)
    set(FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP 1)
  endif()

  if(NOT DEFINED FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_THREADS)
    set(FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_THREADS 1)
  endif()

  set(assert_mpi_file
      ${CMAKE_CURRENT_BINARY_DIR}/assert_mpi_${TESTNAME}_${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP}.cpp
      )
  configure_file(${PROJECT_SOURCE_DIR}/unittests/assert_mpi.cpp.in ${assert_mpi_file})

  add_executable(
    ${TESTNAME}
    ${PROJECT_SOURCE_DIR}/unittests/4C_gtest_main_mpi_test.cpp
    ${assert_mpi_file}
    ${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_SOURCE}
    )
  # Store unit test executables directly inside the tests/ directory
  set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

  # Do not try to build tests as unity files.
  set_target_properties(${TESTNAME} PROPERTIES UNITY_BUILD OFF)

  four_c_set_up_executable(${TESTNAME})

  # All libraries are linked as PRIVATE since a unit test executable cannot be used as a dependency itself.
  target_link_libraries(${TESTNAME} PRIVATE gtest gmock)

  # Link to common helpers for unit tests
  target_link_libraries(${TESTNAME} PRIVATE unittests_common)

  # the first process will write a unit test report
  separate_arguments(
    MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST UNIX_COMMAND ${MPIEXEC_EXTRA_OPTS_FOR_TESTING}
    )

  set(mpi_arguments
      ${MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST}
      -np
      1
      $<TARGET_FILE:${TESTNAME}>
      --gtest_output=xml:unittest_reports/${TESTNAME}_report.xml
      )
  # if there is more than one process, spawn the remaining ones without a report
  if(FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP GREATER "1")
    math(EXPR remaining_procs "${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP}-1")
    list(
      APPEND
      mpi_arguments
      :
      ${MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST}
      -np
      ${remaining_procs}
      $<TARGET_FILE:${TESTNAME}>
      )
  endif()

  # Calculate the total number of processors required
  math(
    EXPR
    TOTAL_NUM_PROCESSORS
    "${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP}*${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_THREADS}"
    )

  add_test(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${mpi_arguments})
  set_tests_properties(${TESTNAME} PROPERTIES TIMEOUT ${UNITTEST_TIMEOUT} LABELS minimal)
  set_tests_properties(${TESTNAME} PROPERTIES PROCESSORS ${TOTAL_NUM_PROCESSORS})
  set_tests_properties(
    ${TESTNAME}
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_THREADS}"
    )

  # Deal with additional support files
  set(FOUR_C_TEST_SUPPORT_FILE_DIR "${CMAKE_BINARY_DIR}/tests/support_files/${TESTNAME}/")

  foreach(support_file ${FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_SUPPORT_FILES})
    # Get file name relative to current list file
    cmake_path(RELATIVE_PATH support_file)
    message(
      DEBUG
      "Copying support file ${support_file} to ${FOUR_C_TEST_SUPPORT_FILE_DIR}/${support_file}"
      )

    # Only substitute CMake variables with @ syntax. The $ style variables we use internally are not substituted.
    configure_file(${support_file} ${FOUR_C_TEST_SUPPORT_FILE_DIR}/${support_file} @ONLY)
  endforeach()
  target_compile_definitions(
    ${TESTNAME} PRIVATE -DFOUR_C_TEST_SUPPORT_FILE_DIR="${FOUR_C_TEST_SUPPORT_FILE_DIR}"
    )

  add_dependencies(unittests ${TESTNAME})

  message(DEBUG "Picked up unit tests ${TESTNAME}")
endfunction()
