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
#   TEST_FILE:                    Name of the input file in the directory tests/performance_tests. If this name
#                                 ends in `.in`, the file is treated as a template and the suffix is stripped
#                                 for the generated test name and the file actually handed to 4C. Use the `.in`
#                                 suffix for input files that contain placeholders (e.g. @NUM_NODES@) that are
#                                 not valid 4C-YAML input and would fail schema validation.
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
#   PLACEHOLDERS:                 Additional placeholders to substitute in the input file, on top of the mandatory
#                                 @MESH_FILE@ substitution. One quoted string per placeholder: "NAME MINIMAL_VALUE
#                                 FULL_VALUE". Every @NAME@ in the input file becomes MINIMAL_VALUE (minimal variant)
#                                 or FULL_VALUE (full variant). Example:
#                                   PLACEHOLDERS
#                                   "NUM_NODES      192 134535"
#                                   "NUM_ELEMENTS   191 134534"
#                                 Use for problem-specific parameters that depend on the mesh/data variant, without
#                                 teaching this generic function about any physics module's parameters.
#                                 Each entry needs exactly 3 tokens, a valid/unique NAME that actually occurs as
#                                 @NAME@ in the resolved test file. NAME "MESH_FILE" is reserved (it would collide
#                                 with the mandatory @MESH_FILE@ substitution) and is rejected.
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
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES COPY_FILES PLACEHOLDERS)
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

  string(REGEX REPLACE "\\.in$" "" _generated_test_file_name "${_parsed_TEST_FILE}")

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

  # Validate PLACEHOLDERS entries: 3 tokens (NAME MINIMAL_VALUE FULL_VALUE), NAME valid and unique,
  # and NAME must occur as @NAME@ in the test file source.
  if(_parsed_PLACEHOLDERS)
    file(READ "${test_file_full_path}" _test_file_source_content)
    set(_seen_placeholder_names "")

    foreach(_placeholder_entry IN LISTS _parsed_PLACEHOLDERS)
      string(REGEX MATCHALL "[^ \t]+" _placeholder_tokens "${_placeholder_entry}")
      list(LENGTH _placeholder_tokens _num_placeholder_tokens)
      if(NOT _num_placeholder_tokens EQUAL 3)
        message(
          FATAL_ERROR
            "Invalid PLACEHOLDERS entry '${_placeholder_entry}' for test file ${test_file_full_path}: expected exactly 3 whitespace-separated tokens \"NAME MINIMAL_VALUE FULL_VALUE\" as one quoted argument, got ${_num_placeholder_tokens}."
          )
      endif()
      list(GET _placeholder_tokens 0 _placeholder_name)

      if(NOT _placeholder_name MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(
          FATAL_ERROR
            "Invalid PLACEHOLDERS name '${_placeholder_name}' for test file ${test_file_full_path}: must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]*."
          )
      endif()

      if(_placeholder_name STREQUAL "MESH_FILE")
        message(
          FATAL_ERROR
            "PLACEHOLDERS name 'MESH_FILE' is reserved for the mandatory mesh substitution; choose a different name."
          )
      endif()

      if(_placeholder_name IN_LIST _seen_placeholder_names)
        message(
          FATAL_ERROR
            "Duplicate PLACEHOLDERS name '${_placeholder_name}' for test file ${test_file_full_path}: each name must only be given once."
          )
      endif()
      list(APPEND _seen_placeholder_names "${_placeholder_name}")

      string(FIND "${_test_file_source_content}" "@${_placeholder_name}@" _placeholder_position)
      if(_placeholder_position EQUAL -1)
        message(
          FATAL_ERROR
            "PLACEHOLDERS name '${_placeholder_name}' does not occur as @${_placeholder_name}@ in test file source ${test_file_full_path}."
          )
      endif()
    endforeach()
  endif()

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

  set(name_of_test ${_generated_test_file_name}-performance)
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

  # configure the respective input file for the test (exchange the MESH_FILE placeholder and any
  # additional placeholders given via PLACEHOLDERS)
  set(configured_input_file "${test_directory}/${_generated_test_file_name}")
  set(_sed_expression "s|@MESH_FILE@|${mesh_file_full_path}|g")

  foreach(_placeholder_entry IN LISTS _parsed_PLACEHOLDERS)
    string(REGEX MATCHALL "[^ \t]+" _placeholder_tokens "${_placeholder_entry}")
    list(GET _placeholder_tokens 0 _placeholder_name)
    list(GET _placeholder_tokens 1 _placeholder_minimal_value)
    list(GET _placeholder_tokens 2 _placeholder_full_value)

    if(FOUR_C_ENABLE_FULL_PERFORMANCE_TESTS)
      set(_placeholder_value "${_placeholder_full_value}")
    else()
      set(_placeholder_value "${_placeholder_minimal_value}")
    endif()

    string(APPEND _sed_expression ";s|@${_placeholder_name}@|${_placeholder_value}|g")
  endforeach()

  set(_configure_inputfile
      "sed '${_sed_expression}' ${test_file_full_path} > ${configured_input_file}"
      )

  # Safety net: after substitution, no @NAME@-shaped token must remain in the configured file.
  set(_check_no_leftover_placeholders
      "if grep -nE '@[A-Za-z_][A-Za-z0-9_]*@' ${configured_input_file}; then \
echo 'Error: unsubstituted placeholder(s) remain in ${configured_input_file}; check MESH_FILE/PLACEHOLDERS for ${_parsed_TEST_FILE}.'; \
exit 1; \
fi"
      )

  # define the run command
  set(test_command
      "mkdir -p ${test_directory} \
                && ${_configure_inputfile} \
                && ${_check_no_leftover_placeholders} \
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
