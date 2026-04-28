# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

###------------------------------------------------------------------ Performance test
# A central function to define a performance test. The test consists of two variants. During normal
# testing, a minimal version of the test is executed to check the functionality of the test. During
# actual performance testing, a full version of the test is executed. Performance data is collected
# in both scenarios, but only in the full testing it is saved for visualization. Full testing can be
# enabled by enabling the option FOUR_C_ENABLE_FULL_PERFORMANCE_TESTS.
#
# required parameters:
#   TEST_FILE:                    Name of the input file in the directory tests/performance_tests.
#   MESH:                         Path the mesh file in the directory tests/performance_tests/meshes/{minimal/full}/. The minimal mesh is used during regular testing, and the full mesh is used for performance testing.
#
# optional parameters:
#   NP_FULL:                      Number of processors the test should use. Fallback to all available ranks if not specified.
#   NP_MINIMAL:                   Number of processors the test should use. Fallback to 1 if not specified.
#   TIMEOUT_MINIMAL:              Manually defined duration for test timeout for the minimal test; defaults to global timeout if not specified.
#   TIMEOUT_FULL:                 Manually defined duration for test timeout for the full test; defaults to 10 minutes if not specified.
#   LABELS:                       Add labels to the test; The label `performance_tests` is added by default.
#   REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                 Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                 The supported version constraint operators are: >=, <=, >, <, ==
#                                 If multiple dependencies are provided, all must be met for the test to run.
#                                 Note that the version is the _internal_ version that 4C assigns to the dependency.
#   COPY_FILES:                   List of files that should be copied to the build directory for the test.
function(four_c_performance_test)
  set(options "")
  set(oneValueArgs
      TEST_FILE
      MESH
      NP_FULL
      NP_MINIMAL
      TIMEOUT_MINIMAL
      TIMEOUT_FULL
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES COPY_FILES)
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  # validate input arguments
  if(DEFINED _parsed_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "There are unparsed arguments: ${_parsed_UNPARSED_ARGUMENTS}!")
  endif()

  assert_required_arguments(_parsed TEST_FILE MESH)

  if(NOT DEFINED _parsed_NP_FULL)
    # Query the system for the number of logical cores
    cmake_host_system_information(RESULT _parsed_NP_FULL QUERY NUMBER_OF_LOGICAL_CORES)

    if(NOT _parsed_NP_FULL OR _parsed_NP_FULL EQUAL 0)
      message(
        FATAL_ERROR
          "Could not determine the number of logical cores on the system. Please specify the number of processors for the full performance test using the NP_FULL argument for each test."
        )
    endif()
  endif()

  if(NOT DEFINED _parsed_NP_MINIMAL)
    set(_parsed_NP_MINIMAL 1)
  endif()
  if(_parsed_NP_MINIMAL GREATER 3)
    message(
      FATAL_ERROR
        "Number of processors for minimal performance tests must be less than or equal to 3!"
      )
  endif()

  # Full file to the input file
  set(test_file_full_path "${PROJECT_SOURCE_DIR}/tests/performance_tests/${_parsed_TEST_FILE}")
  if(NOT EXISTS ${test_file_full_path})
    message(
      FATAL_ERROR "Test source file ${test_file_full_path} of the performance test does not exist!"
      )
  endif()

  # We need to reconfigure if we change the input file
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${test_file_full_path}"
    )

  # Full path to the both mesh files
  set(minimal_mesh_file_full_path
      "${PROJECT_SOURCE_DIR}/tests/performance_tests/meshes/minimal/${_parsed_MESH}"
      )
  set(full_mesh_file_full_path
      "${PROJECT_SOURCE_DIR}/tests/performance_tests/meshes/full/${_parsed_MESH}"
      )

  # check if both mesh files exist
  if(NOT EXISTS ${minimal_mesh_file_full_path})
    message(
      FATAL_ERROR
        "Minimal mesh file ${_parsed_MESH} of the performance test does not exist! Expected location: ${minimal_mesh_file_full_path}. Note: Performance tests need to provide a minimal and a full version of the mesh."
      )
  endif()
  if(NOT EXISTS ${full_mesh_file_full_path})
    message(
      FATAL_ERROR
        "Full mesh file ${_parsed_MESH} of the performance test does not exist! Expected location: ${full_mesh_file_full_path}. Note: Performance tests need to provide a minimal and a full version of the mesh."
      )
  endif()

  set(name_of_test ${_parsed_TEST_FILE}-performance)
  set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/performance_tests/${name_of_test})

  # copy additional files to the test directory
  if(_parsed_COPY_FILES)
    foreach(_file_name IN LISTS _parsed_COPY_FILES)
      if(NOT EXISTS ${_file_name})
        message(FATAL_ERROR "File ${_file_name} does not exist!")
      endif()

      list(APPEND _run_copy_files "cp ${_file_name} ${test_directory}")
    endforeach()

    list(JOIN _run_copy_files " && " _run_copy_files)
  else()
    # no-op command to do nothing
    set(_run_copy_files ":")
  endif()

  # configure the respective input test with the respective mesh
  if(FOUR_C_ENABLE_FULL_PERFORMANCE_TESTS)
    set(mesh_file_full_path ${full_mesh_file_full_path})

    if("${_parsed_TIMEOUT_FULL}" STREQUAL "")
      # default timeout for full performance tests is 10 minutes.
      set(_parsed_TIMEOUT_FULL 600)
    endif()
    set(timeout "${_parsed_TIMEOUT_FULL}")
    set(num_procs "${_parsed_NP_FULL}")
  else()
    set(mesh_file_full_path ${minimal_mesh_file_full_path})

    # No special default for the minimal test timeout (we get the default from the global timeout)
    set(timeout "${_parsed_TIMEOUT_MINIMAL}")
    set(num_procs "${_parsed_NP_MINIMAL}")
  endif()

  # configure the respective input file for the test (exchange the MESH_FILE placeholder)
  set(configured_input_file "${test_directory}/${_parsed_TEST_FILE}")
  set(_configure_inputfile
      "sed 's|@MESH_FILE@|${mesh_file_full_path}|g' ${test_file_full_path} > ${configured_input_file}"
      )

  # define the run command
  set(test_command
      "mkdir -p ${test_directory} \
                && ${_configure_inputfile} \
                && ${_run_copy_files} \
                && ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${num_procs} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${configured_input_file} ${test_directory}/xxx"
      )

  # Add performance_tests label
  list(APPEND _parsed_LABELS "performance_tests")

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    CLEANUP_FIXTURES
    collect_performance_test_results
    TOTAL_PROCS
    ${num_procs}
    TIMEOUT
    "${timeout}"
    LABELS
    "${_parsed_LABELS}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
endfunction()

###------------------------------------------------------------------ Performance test
# A central function to collect the results of the performance tests and save them in a json file.
#
# required parameters:
#   TARGET_FILE:                  Path to the json file where the results should be saved.
#
# optional parameters:
#   ALLOW_EMPTY:                  If this label is set, the collection will not fail if no performance test results are found.
function(four_c_collect_performance_test_results)
  set(options ALLOW_EMPTY)
  set(oneValueArgs TARGET_FILE)
  set(multiValueArgs "")
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  # validate input arguments
  if(DEFINED _parsed_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "There are unparsed arguments: ${_parsed_UNPARSED_ARGUMENTS}!")
  endif()

  assert_required_arguments(_parsed TARGET_FILE)

  set(name_of_test performance_test_results_collection)

  if("${_parsed_TARGET_FILE}" STREQUAL "")
    message(FATAL_ERROR "The TARGET_FILE argument must not be empty.")
  endif()

  set(allow_empty_flag "")
  if(_parsed_ALLOW_EMPTY)
    set(allow_empty_flag "--allow-empty")
  endif()

  set(test_command
      "collect-performance-test-results ${PROJECT_BINARY_DIR}/framework_test_output/performance_tests ${_parsed_TARGET_FILE} ${allow_empty_flag}"
      )

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    TOTAL_PROCS
    1
    REQUIRED_DEPENDENCIES
    "Python"
    )

  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_CLEANUP collect_performance_test_results)
endfunction()
