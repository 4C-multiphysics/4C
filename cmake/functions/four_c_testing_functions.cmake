# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

###------------------------------------------------------------------ Helper Functions

# define this test (name_of_test) as a "setup fixture" (named name_of_fixture). Other tests may
# add a dependency on such a setup fixture through the require_fixture() function. CTest will
# then ensure that a) required fixtures are included if necessary and b) tests are executed in
# the correct order.
#
# In the 4C test suite we need such dependencies between a test e.g. for an initial simulation
# and a test for a restart, or between an initial simulation and a post-processing test.
function(define_setup_fixture name_of_test name_of_fixture)
  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_SETUP ${name_of_fixture})
endfunction()

# add a required test (name_of_required_test) to this test (name_of_test). The required
# test must be defined through define_setup_fixture() as "setup fixture". For more details on why
# these functions are needed, have a look at the documentation of define_setup_fixture().
function(require_fixture name_of_test name_of_required_test)
  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_REQUIRED "${name_of_required_test}")
endfunction()

function(set_environment name_of_test)
  if(FOUR_C_WITH_PYTHON)
    # Add Python virtual environment
    if(NOT DEFINED FOUR_C_PYTHON_VENV_BUILD OR "${FOUR_C_PYTHON_VENV_BUILD}" STREQUAL "")
      message(FATAL_ERROR "Could not determine Python virtual environment for testing.")
    endif()

    # We also need to add the PYTHONPATH with the site-packages of the virtual environment
    if(NOT DEFINED FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES
       OR "${FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES}" STREQUAL ""
       )
      message(
        FATAL_ERROR
          "Could not determine site-packages directory of Python virtual environment for testing."
        )
    endif()

    set_property(
      TEST ${name_of_test}
      APPEND
      PROPERTY ENVIRONMENT_MODIFICATION
               "PATH=path_list_prepend:${FOUR_C_PYTHON_VENV_BUILD}/bin"
               "PYTHONPATH=path_list_prepend:${FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES}"
      )
  endif()
endfunction()

# set fail expressions to this test (name_of_test)
function(set_fail_expression name_of_test)
  set_tests_properties(${name_of_test} PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR:; ERROR ;Error ")
endfunction()

# set label to this test (name_of_test)
function(set_label name_of_test label)
  set_tests_properties(${name_of_test} PROPERTIES LABELS ${label})
endfunction()

# set number of processors (num_proc) to this test (name_of_test)
function(set_processors name_of_test num_proc)
  set_tests_properties(${name_of_test} PROPERTIES PROCESSORS ${num_proc})
endfunction()

# set this test(name_of_test) to run in serial
function(set_run_serial name_of_test)
  set_tests_properties(${name_of_test} PROPERTIES RUN_SERIAL TRUE)
endfunction()

# set timeout to this test (name_of_test). Optional, set timeout value
function(set_timeout name_of_test)
  if("${ARGN}" STREQUAL "")
    set_tests_properties(${name_of_test} PROPERTIES TIMEOUT ${FOUR_C_TEST_GLOBAL_TIMEOUT})
  else()
    set_tests_properties(${name_of_test} PROPERTIES TIMEOUT ${ARGN})
  endif()
endfunction()

# Mark a test as skipped. This registers a dummy test printing an informational message.
# Running ctest will display that test as skipped, to make it clear that this test exists
# but is not executed. The reason may be explained in the message.
function(skip_test name_of_test message)
  # The dummy test needs to report a arbitrary error code that ctest interprets as "skipped".
  set(dummy_command "echo \"${message}\"; exit 42")
  add_test(NAME ${name_of_test} COMMAND bash -c "${dummy_command}")
  set_tests_properties(${name_of_test} PROPERTIES SKIP_RETURN_CODE 42)
endfunction()

# Extract all required dependencies from the
# previous test and add it to the required dependencies list
function(append_required_dependencies_from previous_test list_of_dependencies)
  get_test_property(${previous_test} _internal_REQUIRED_DEPENDENCIES previous_dependencies)
  if(previous_dependencies AND NOT previous_dependencies STREQUAL "NOTFOUND")
    list(APPEND ${list_of_dependencies} ${previous_dependencies})
    set(${list_of_dependencies}
        "${${list_of_dependencies}}"
        PARENT_SCOPE
        )
  endif()
endfunction()

# Check that required dependencies are met. If not, a skip message is built up and returned in
# the variable 'result'. If the skip message is not empty after calling this function, the test
# should be skipped.
function(check_required_dependencies result requirements)
  foreach(dep IN LISTS requirements)
    # Check if this is a dependency with version
    if(dep MATCHES "^([^><=]+)([><=]+)([0-9\\.]+)$")
      set(dep_name "${CMAKE_MATCH_1}")
      set(dep_version "${CMAKE_MATCH_3}")
      if(CMAKE_MATCH_2 STREQUAL ">=")
        set(dep_version_constraint "VERSION_GREATER_EQUAL")
      elseif(CMAKE_MATCH_2 STREQUAL "<=")
        set(dep_version_constraint "VERSION_LESS_EQUAL")
      elseif(CMAKE_MATCH_2 STREQUAL ">")
        set(dep_version_constraint "VERSION_GREATER")
      elseif(CMAKE_MATCH_2 STREQUAL "<")
        set(dep_version_constraint "VERSION_LESS")
      elseif(CMAKE_MATCH_2 STREQUAL "==")
        set(dep_version_constraint "VERSION_EQUAL")
      else()
        message(
          FATAL_ERROR "Unsupported version constraint '${CMAKE_MATCH_2}' for dependency '${dep}'"
          )
      endif()
    else()
      set(dep_name "${dep}")
      unset(dep_version)
      unset(dep_version_constraint)
    endif()

    four_c_sanitize_package_name(${dep_name} dep_sanitized)

    # Check that we even know this dependency
    if(NOT DEFINED "FOUR_C_WITH_${dep_sanitized}")
      message(FATAL_ERROR "Unknown dependency ${dep_name} requested for test")
    endif()

    if(NOT FOUR_C_WITH_${dep_sanitized})
      string(APPEND skip_message "Skipping because FOUR_C_WITH_${dep_sanitized} is not enabled.\n")
    elseif(DEFINED dep_version)
      if(NOT DEFINED "FOUR_C_${dep_sanitized}_INTERNAL_VERSION")
        message(FATAL_ERROR "Requiring '${dep}' but no internal version is known for ${dep_name}")
      endif()

      set(actual_version "${FOUR_C_${dep_sanitized}_INTERNAL_VERSION}")
      if(NOT actual_version ${dep_version_constraint} ${dep_version})
        string(
          APPEND
          skip_message
          "Skipping because ${dep_name} version ${actual_version} does not satisfy constraint ${dep}.\n"
          )
      endif()
    endif()
  endforeach()
  set(${result}
      "${skip_message}"
      PARENT_SCOPE
      )
endfunction()

# Helper to check that required arguments are provided in parsed cmake_parse_arguments
function(assert_required_arguments prefix)
  set(required_args ${ARGN})
  foreach(arg IN LISTS required_args)
    if(NOT DEFINED ${prefix}_${arg})
      message(SEND_ERROR "Argument '${arg}' is required but not provided!")
      set(_had_error TRUE)
    endif()
  endforeach()
  if(DEFINED _had_error)
    message(FATAL_ERROR "One or more required arguments were not provided!")
  endif()
endfunction()

# Check whether a test with given name exists
# Note that this will only work for tests defined through our own testing functions
function(check_test_exists result name)
  get_test_property(${name} _internal_INPUT_FILE _check)
  if(_check)
    set(${result}
        TRUE
        PARENT_SCOPE
        )
  else()
    set(${result}
        FALSE
        PARENT_SCOPE
        )
  endif()
endfunction()

# Internal helper that adds a test to the 4C test suite. Do not call this directly.
function(_add_test_with_options)
  set(options "")
  set(oneValueArgs
      NAME_OF_TEST
      ADDITIONAL_FIXTURE
      TOTAL_PROCS
      TIMEOUT
      INPUT_FILE
      OUTPUT_DIR
      )
  set(multiValueArgs TEST_COMMAND REQUIRED_DEPENDENCIES LABELS CLEANUP_FIXTURES)
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  if(DEFINED _parsed_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "There are unparsed arguments: ${_parsed_UNPARSED_ARGUMENTS}!")
  endif()

  assert_required_arguments(_parsed NAME_OF_TEST TEST_COMMAND)

  if(NOT DEFINED _parsed_CLEANUP_FIXTURES)
    set(_parsed_CLEANUP_FIXTURES "")
  endif()

  if(NOT DEFINED _parsed_ADDITIONAL_FIXTURE)
    set(_parsed_ADDITIONAL_FIXTURE "")
  endif()

  if(NOT DEFINED _parsed_TOTAL_PROCS)
    set(_parsed_TOTAL_PROCS 1)
  endif()

  if(NOT DEFINED _parsed_TIMEOUT)
    set(_parsed_TIMEOUT "")
  endif()

  if(NOT DEFINED _parsed_LABELS)
    set(_parsed_LABELS "")
  endif()

  # ensure that the cleanup fixture is added
  list(APPEND _parsed_CLEANUP_FIXTURES "test_cleanup")

  # add all required dependencies from the additional fixture, if specified
  if(NOT _parsed_ADDITIONAL_FIXTURE STREQUAL "")
    append_required_dependencies_from(${_parsed_ADDITIONAL_FIXTURE} _parsed_REQUIRED_DEPENDENCIES)
  endif()

  # check if all required dependencies are present
  check_required_dependencies(skip_message "${_parsed_REQUIRED_DEPENDENCIES}")

  if(NOT skip_message STREQUAL "")
    # The dummy test needs to report a arbitrary error code that ctest interprets as "skipped".
    set(dummy_command "echo \"${skip_message}\"; exit 42")
    # Add a dummy test that just prints the skip message instead of the real test
    add_test(NAME ${_parsed_NAME_OF_TEST} COMMAND bash -c "${dummy_command}")
    set_tests_properties(${_parsed_NAME_OF_TEST} PROPERTIES SKIP_RETURN_CODE 42)
    message(VERBOSE "Skipping test ${_parsed_NAME_OF_TEST}: ${skip_message}")

    # Note: Cleanup fixtures are not required for skipped tests
    require_fixture(${_parsed_NAME_OF_TEST} "${_parsed_ADDITIONAL_FIXTURE}")
  else()
    # Add the real test
    add_test(NAME ${_parsed_NAME_OF_TEST} COMMAND bash -c "${_parsed_TEST_COMMAND}")

    require_fixture(
      ${_parsed_NAME_OF_TEST} "${_parsed_ADDITIONAL_FIXTURE};${_parsed_CLEANUP_FIXTURES}"
      )
  endif()

  set_environment(${_parsed_NAME_OF_TEST})
  set_processors(${_parsed_NAME_OF_TEST} ${_parsed_TOTAL_PROCS})
  define_setup_fixture(${_parsed_NAME_OF_TEST} ${_parsed_NAME_OF_TEST})
  set_timeout(${_parsed_NAME_OF_TEST} ${_parsed_TIMEOUT})

  if(NOT ${_parsed_LABELS} STREQUAL "")
    set_label(${_parsed_NAME_OF_TEST} ${_parsed_LABELS})
  endif()

  # Set a few properties that we use internally to build up more complex test chains.
  # These properties are not used or understood by CMake itself.
  if(DEFINED _parsed_INPUT_FILE)
    set_tests_properties(
      ${_parsed_NAME_OF_TEST} PROPERTIES _internal_INPUT_FILE ${_parsed_INPUT_FILE}
      )
  endif()
  if(DEFINED _parsed_OUTPUT_DIR)
    set_tests_properties(
      ${_parsed_NAME_OF_TEST} PROPERTIES _internal_OUTPUT_DIR ${_parsed_OUTPUT_DIR}
      )
  endif()
  set_tests_properties(
    ${_parsed_NAME_OF_TEST}
    PROPERTIES _internal_REQUIRED_DEPENDENCIES "${_parsed_REQUIRED_DEPENDENCIES}"
    )

endfunction()

##
# Central function to define a 4C test based on an input file
#
# Use this function to define a new test in tests/lists_of_tests.cmake. You may use
# further four_c_test_* functions to define dependent tests based on this test. In this case,
# you can use the RETURN_AS option to store the name of the defined test in a variable for later
# use. The other four_c_test_* functions take this information in the BASED_ON parameter.
#
# required parameters:
#   TEST_FILE:                    must equal the name of an input file in directory tests/input_files
#
# optional parameters:
#   NP:                           Number of processors the test should use. Fallback to 1 if not specified.
#   TIMEOUT:                      Manually defined duration for test timeout; defaults to global timeout if not specified
#   OMP_THREADS:                  Number of OpenMP threads per processor the test should use; defaults to no OpenMP if not specified
#   LABELS:                       Add labels to the test
#   REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                 Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                 The supported version constraint operators are: >=, <=, >, <, ==
#                                 If multiple dependencies are provided, all must be met for the test to run.
#                                 Note that the version is the _internal_ version that 4C assigns to the dependency.
#   RETURN_AS:                    A variable name that allows to add further dependent tests based on this test.
function(four_c_test)
  set(options "")
  set(oneValueArgs
      TEST_FILE
      NP
      TIMEOUT
      OMP_THREADS
      RETURN_AS
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(_parsed TEST_FILE)

  if(NOT DEFINED _parsed_NP)
    set(_parsed_NP 1)
  endif()
  if(_parsed_NP GREATER 3)
    message(FATAL_ERROR "Number of processors must be less than or equal to 3!")
  endif()

  if(NOT DEFINED _parsed_OMP_THREADS)
    set(_parsed_OMP_THREADS 0)
  endif()

  # check if source files exist
  set(test_file_full_path "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_TEST_FILE}")

  if(NOT EXISTS ${test_file_full_path})
    message(FATAL_ERROR "Test source file ${file_name} does not exist!")
  endif()

  set(name_of_test ${_parsed_TEST_FILE}-p${_parsed_NP})
  set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})

  # Optional OpenMP threads per processor
  set(total_procs ${_parsed_NP})

  if(${_parsed_OMP_THREADS})
    set(name_of_test ${name_of_test}-OMP${_parsed_OMP_THREADS})
    set(test_command
        "export OMP_NUM_THREADS=${_parsed_OMP_THREADS} && ${test_command} && unset OMP_NUM_THREADS"
        )
    math(EXPR total_procs "${_parsed_NP}*${_parsed_OMP_THREADS}")
  endif()

  # Final test name is determined at this point, so we can set it as "return value" if requested
  if(DEFINED _parsed_RETURN_AS)
    set(${_parsed_RETURN_AS}
        ${name_of_test}
        PARENT_SCOPE
        )
  endif()

  set(test_command
      "mkdir -p ${test_directory} \
                && ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${test_file_full_path} ${test_directory}/xxx"
      )

  # Optional timeout
  if(DEFINED _parsed_TIMEOUT)
    # scale testtimeout with the global test timeout scale
    math(EXPR _parsed_TIMEOUT "${FOUR_C_TEST_TIMEOUT_SCALE} * ${_parsed_TIMEOUT}")
  endif()

  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    TOTAL_PROCS
    ${total_procs}
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    INPUT_FILE
    "${test_file_full_path}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
endfunction()

##
# Define a restart test that restarts from a previous test
#
# required parameters:
#   BASED_ON:                name of the base test that created the restart files
#   RESTART_STEP:            number of the restart step to restart from or last_possible
#   SAME_FILE or TEST_FILE:  either SAME_FILE to indicate that the restart should be done from the same input file
#                            as the base test, or TEST_FILE to indicate that a different input file should be used for the restart
#
# optional parameters:
#   NP:                      number of processors the test should use. Fallback to 1 if not specified.
#   TIMEOUT:                 manually defined duration for test timeout; defaults to global timeout if not specified
#   OMP_THREADS:             number of OpenMP threads per processor the test should use; defaults to no OpenMP if not specified
#   LABELS:                  Add labels to the test
#   REQUIRED_DEPENDENCIES:   Any required external dependencies. The test will be skipped if the dependencies are not met.
#   RETURN_AS:               A variable name that allows to add further dependent tests based on this test.
function(__four_c_test_restart)
  set(options SAME_FILE)
  set(oneValueArgs
      BASED_ON
      TEST_FILE
      NP
      RESTART_STEP
      TIMEOUT
      OMP_THREADS
      RETURN_AS
      ASSERT_RESTART_STEP
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(_parsed BASED_ON RESTART_STEP)

  if(NOT DEFINED _parsed_NP)
    set(_parsed_NP 1)
  endif()
  if(_parsed_NP GREATER 3)
    message(FATAL_ERROR "Number of processors must be less than or equal to 3!")
  endif()

  if(NOT DEFINED _parsed_OMP_THREADS)
    set(_parsed_OMP_THREADS 0)
  endif()

  if(parsed_SAME_FILE AND DEFINED _parsed_TEST_FILE)
    message(FATAL_ERROR "You cannot specify both SAME_FILE and TEST_FILE")
  endif()
  if(NOT _parsed_SAME_FILE AND NOT DEFINED _parsed_TEST_FILE)
    message(FATAL_ERROR "You must specify either SAME_FILE or TEST_FILE")
  endif()

  # In case we reuse the same file as the base test, get the input file from there
  if(_parsed_SAME_FILE)
    # Get or initialize restart counter for this base test
    get_property(_restart_count GLOBAL PROPERTY ${_parsed_BASED_ON}_RESTART_COUNT)
    if(NOT DEFINED _restart_count OR _restart_count STREQUAL "")
      set(_restart_count 0)
    endif()

    # Increment counter
    math(EXPR _restart_count "${_restart_count} + 1")
    set_property(GLOBAL PROPERTY ${_parsed_BASED_ON}_RESTART_COUNT ${_restart_count})

    set(name_of_test "${_parsed_BASED_ON}-restart_${_parsed_RESTART_STEP}-p${_parsed_NP}")
    get_test_property(${_parsed_BASED_ON} _internal_INPUT_FILE test_file_full_path)
    get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)
    set(restart_arguments "--restart=${_parsed_RESTART_STEP}")
  else()
    # Restart from a different testfile
    set(name_of_test
        "${_parsed_BASED_ON}-restart_${_parsed_RESTART_STEP}-with_${_parsed_TEST_FILE}-p${_parsed_NP}"
        )
    set(test_file_full_path "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_TEST_FILE}")
    set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
    get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR base_directory)
    set(restart_arguments "--restartfrom=${base_directory}/xxx --restart=${_parsed_RESTART_STEP}")

    if(NOT EXISTS ${test_file_full_path})
      message(FATAL_ERROR "Test source file ${test_file_full_path} does not exist")
    endif()
  endif()

  # Basic test command
  set(test_command
      "mkdir -p ${test_directory} \
                && ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${test_file_full_path} ${test_directory}/xxx ${restart_arguments}"
      )

  # Possibly enhanced with OpenMP
  set(total_procs ${_parsed_NP})
  if(${_parsed_OMP_THREADS})
    set(name_of_test ${name_of_test}-OMP${_parsed_OMP_THREADS})
    set(test_command
        "export OMP_NUM_THREADS=${_parsed_OMP_THREADS} && ${test_command} && unset OMP_NUM_THREADS"
        )
    math(EXPR total_procs "${_parsed_NP}*${_parsed_OMP_THREADS}")
  endif()

  if(DEFINED _parsed_RETURN_AS)
    set(${_parsed_RETURN_AS}
        ${name_of_test}
        PARENT_SCOPE
        )
  endif()

  check_test_exists(_base_test_exists ${_parsed_BASED_ON})
  if(NOT _base_test_exists)
    message(FATAL_ERROR "Base test ${_parsed_BASED_ON} for restart does not exist.")
  endif()

  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    ADDITIONAL_FIXTURE
    ${_parsed_BASED_ON}
    TOTAL_PROCS
    "${total_procs}"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    INPUT_FILE
    "${test_file_full_path}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )

  # If ASSERT_RESTART_STEP is specified, verify the restart step in the control file
  if(DEFINED _parsed_ASSERT_RESTART_STEP)
    # Determine control file name based on restart type
    if(_parsed_SAME_FILE)
      set(control_file "${test_directory}/xxx-${_restart_count}.control")
    else()
      set(control_file "${test_directory}/xxx.control")
    endif()

    set(name_of_restart_check "${name_of_test}-check_restart_step")
    set(check_command "check-restart-step ${control_file} ${_parsed_ASSERT_RESTART_STEP}")

    # Ensure that Python is listed as required dependency
    list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")
    _add_test_with_options(
      NAME_OF_TEST
      ${name_of_restart_check}
      TEST_COMMAND
      ${check_command}
      ADDITIONAL_FIXTURE
      ${name_of_test}
      LABELS
      "${_parsed_LABELS}"
      REQUIRED_DEPENDENCIES
      "${_parsed_REQUIRED_DEPENDENCIES}"
      )
  endif()
endfunction()

##
# Define a comparison test that compares a csv or yaml result file created by a test against
# an expected reference file
#
# required parameters:
#   BASED_ON:              name of the base test that created the result file to compare against
#   RESULT_FILE:           name of the result file created by the base test
#   REFERENCE_FILE:        name of the reference file to compare against (must be located in tests/input_files)
#   TOL_R:                 relative tolerance for comparison
#   TOL_A:                 absolute tolerance for comparison
#
# optional parameters:
#   LABELS:                add labels to the test
#   REQUIRED_DEPENDENCIES: any required external dependencies. The test will be skipped if the dependencies are not met.
#
function(__four_c_test_add_csv_yaml_comparison)
  set(options "")
  set(oneValueArgs
      BASED_ON
      RESULT_FILE
      REFERENCE_FILE
      TOL_R
      TOL_A
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(
    _parsed
    BASED_ON
    RESULT_FILE
    REFERENCE_FILE
    TOL_R
    TOL_A
    )

  set(name_of_csv_comparison_test "${_parsed_BASED_ON}-csv_comparison-${_parsed_RESULT_FILE}")
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  set(csv_comparison_command
      "diff-with-tolerance ${test_directory}/${_parsed_RESULT_FILE} ${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_REFERENCE_FILE} ${_parsed_TOL_R} ${_parsed_TOL_A}"
      )

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_csv_comparison_test}
    TEST_COMMAND
    ${csv_comparison_command}
    ADDITIONAL_FIXTURE
    ${_parsed_BASED_ON}
    LABELS
    "${_parsed_LABELS}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
endfunction()

###------------------------------------------------------------------ Nested Parallelism
# Usage in tests/lists_of_tests.cmake: "four_c_test_nested_parallelism(<name_of_input_file_1> <name_of_input_file_2> <restart_step>)"
# required parameters:
#   TEST_FILE1:                    must equal the name of an input file in directory tests/input_files
#   TEST_FILE2:                    must equal the name of an input file in directory tests/input_files
#
# optional parameters:
#   RESTART_STEP:                 Restart step if restart should be tested.
#   TIMEOUT:                      Manually defined duration for test timeout; defaults to global timeout if not specified
#   LABELS:                       Add labels to the test
#   REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                 Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                 The supported version constraint operators are: >=, <=, >, <, ==
#                                 If multiple dependencies are provided, all must be met for the test to run.
#                                 Note that the version is the _internal_ version that 4C assigns to the dependency.
function(four_c_test_nested_parallelism)
  set(options "")
  set(oneValueArgs TEST_FILE1 TEST_FILE2 RESTART_STEP TIMEOUT)
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(_parsed TEST_FILE1 TEST_FILE2)

  # check if source files exist
  set(test_file_full_path1 "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_TEST_FILE1}")
  set(test_file_full_path2 "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_TEST_FILE2}")

  if(NOT EXISTS ${test_file_full_path1})
    message(FATAL_ERROR "Test source file ${_parsed_TEST_FILE1} does not exist!")
  endif()

  if(NOT EXISTS ${test_file_full_path2})
    message(FATAL_ERROR "Test source file ${_parsed_TEST_FILE2} does not exist!")
  endif()

  set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${_parsed_TEST_FILE1})

  set(name_of_test ${_parsed_TEST_FILE1}-nestedPar)
  set(test_command
      "mkdir -p ${test_directory} &&  ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np 3 $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> --ngroup=2 --glayout=1,2 --nptype=separateInputFiles ${test_file_full_path1} ${test_directory}/xxx ${test_file_full_path2} ${test_directory}/xxxAdditional"
      )

  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    TOTAL_PROCS
    3
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    INPUT_FILE
    "${test_file_full_path1}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )

  if(DEFINED _parsed_RESTART_STEP)
    set(name_of_restart_test ${_parsed_TEST_FILE1}-nestedPar-restart)
    set(restart_test_command
        "mkdir -p ${test_directory} &&  ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np 3 $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> --ngroup=2 --glayout=1,2 --nptype=separateInputFiles ${test_file_full_path1} ${test_directory}/xxx ${test_file_full_path2} ${test_directory}/xxxAdditional --restart=${_parsed_RESTART_STEP},${_parsed_RESTART_STEP}"
        )

    _add_test_with_options(
      NAME_OF_TEST
      ${name_of_restart_test}
      TEST_COMMAND
      ${restart_test_command}
      TOTAL_PROCS
      3
      TIMEOUT
      "${_parsed_TIMEOUT}"
      LABELS
      "${_parsed_LABELS}"
      INPUT_FILE
      "${test_file_full_path1}"
      OUTPUT_DIR
      "${test_directory}"
      ADDITIONAL_FIXTURE
      ${name_of_test}
      REQUIRED_DEPENDENCIES
      "${_parsed_REQUIRED_DEPENDENCIES}"
      )
  endif()
endfunction()

###------------------------------------------------------------------ Tutorial Tests
# Testing a tutorial example
#
# Usage in tests/lists_of_tests.cmake:
#
# four_c_test_tutorial(TEST_FILE <input_file> NP <NP> [COPY_FILES <file1> <file2> ...])"
#
# TEST_FILE: must equal the name of the input file in directory tests/tutorials
# NP: number of MPI ranks for this test
# TIMEOUT: Manually defined duration for test timeout in <seconds>; defaults to global timeout if not specified
# COPY_FILES: copy any additional files to the test directory
# REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                The supported version constraint operators are: >=, <=, >, <, ==
#                                If multiple dependencies are provided, all must be met for the test to run.
#                                Note that the version is the _internal_ version that 4C assigns to the dependency.
function(four_c_test_tutorial)
  set(oneValueArgs TEST_FILE NP TIMEOUT)
  set(multiValueArgs COPY_FILES REQUIRED_DEPENDENCIES)
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

  set(name_of_input_file ${_parsed_TEST_FILE})
  set(num_proc ${_parsed_NP})
  set(name_of_test ${name_of_input_file}-p${num_proc}-fw)
  set(test_directory tutorials/${name_of_input_file})

  # optional timeout (scaled)
  set(local_timeout "")
  if(DEFINED _parsed_TIMEOUT AND NOT "${_parsed_TIMEOUT}" STREQUAL "")
    math(EXPR local_timeout "${FOUR_C_TEST_TIMEOUT_SCALE} * ${_parsed_TIMEOUT}")
  endif()

  list(
    APPEND
    _run_copy_files
    "cp ${PROJECT_SOURCE_DIR}/tests/tutorials/${name_of_input_file} ${test_directory}/xxx.4C.yaml"
    )

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

  set(_run_4C
      ${MPIEXEC_EXECUTABLE}\ ${_mpiexec_all_args_for_testing}\ -np\ ${num_proc}\ $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}>\ ${test_directory}/xxx.4C.yaml\ ${test_directory}/xxx
      ) # 4C is run using the generated input file

  # check whether the dependencies are required
  check_required_dependencies(skip_message "${_parsed_REQUIRED_DEPENDENCIES}")

  if(NOT skip_message STREQUAL "")
    # The dummy test needs to report a arbitrary error code that ctest interprets as "skipped".
    set(dummy_command "echo \"${message}\"; exit 42")
    # Add a dummy test that just prints the skip message instead of the real test
    add_test(NAME ${name_of_test} COMMAND bash -c "${dummy_command}")
    set_tests_properties(${name_of_test} PROPERTIES SKIP_RETURN_CODE 42)
    message(VERBOSE "Skipping test ${name_of_test}: ${_parsed_SKIP_WITH_MESSAGE}")
  else()
    # Add the real test
    add_test(
      NAME ${name_of_test}
      COMMAND
        bash -c
        "mkdir -p ${PROJECT_BINARY_DIR}/${test_directory} && ${_run_copy_files} && ${_run_4C}"
      )
  endif()

  require_fixture(${name_of_test} test_cleanup)
  set_environment(${name_of_test})
  set_fail_expression(${name_of_test})
  set_processors(${name_of_test} ${num_proc})
  set_timeout(${name_of_test} ${local_timeout})
endfunction()

###------------------------------------------------------------------ Cut Tests
# Usage in tests/lists_of_tests.cmake: "four_c_test_cut_test(<num_proc>)"
# <num_proc>: number of processors the test should use
function(four_c_test_cut_test num_proc)
  set(name_of_test test-p${num_proc}-cut)
  set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/cut_test_p${num_proc})

  set(RUNTESTS
      # Run all the cuttests with num_proc except from alex53
      ${MPIEXEC_EXECUTABLE}\ ${_mpiexec_all_args_for_testing}\ -np\ ${num_proc}\ ${PROJECT_BINARY_DIR}/cut_test\ --ignore_test=alex53
      # Run alex53 serially
      ${FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS}\ ${PROJECT_BINARY_DIR}/cut_test\ --test=alex53
      )

  add_test(
    NAME ${name_of_test}
    COMMAND bash -c "mkdir -p ${test_directory} && cd ${test_directory} && ${RUNTESTS}"
    )

  require_fixture(${name_of_test} test_cleanup)
  set_fail_expression(${name_of_test})
  set_environment(${name_of_test})
  set_processors(${name_of_test} ${num_proc})
  set_timeout(${name_of_test})
endfunction()

###------------------------------------------------------------------ Postprocessing Test
# Run ensight postprocessor on previous test
# CAUTION: This tests bases on results of a previous simulation/test
# Usage in tests/lists_of_tests.cmake: "four_c_test_post_processing(<name_of_input_file> <num_proc> <stresstype> <straintype> <startstep> <optional: identifier> <optional: field>"
# <name_of_input_file>: must equal the name of an input file from a previous tests
# <num_proc>: number of processors the test should use
# <num_proc_base_run>: number of processors of precursor base run
# <stresstype>: use post processor with this stresstype
# <straintype>: use post processot with this straintype
# <startstep>: start post processing at this step
# <optional: identifier>: additional identifier that can be added to the test name
# <optional: field>: additional field name that can be added to the test name
function(
  four_c_test_post_processing
  name_of_input_file
  num_proc
  num_proc_base_run
  stresstype
  straintype
  startstep
  )
  set(test_directory
      ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_input_file}-p${num_proc_base_run}
      )

  # set additional output prefix identifier to empty string "" in default case or to specific string if specified as optional input argument
  if(${ARGC} GREATER 7)
    set(IDENTIFIER ${ARGV7})
  else()
    set(IDENTIFIER "")
  endif()

  # set field name to empty string "" in default case or to specific string if specified as optional input argument
  if(${ARGC} GREATER 6)
    set(FIELD ${ARGV6})
  else()
    set(FIELD "")
  endif()

  set(name_of_test "${name_of_input_file}${IDENTIFIER}${FIELD}-p${num_proc}-pp")

  # define macros for serial and parallel runs
  set(RUNPOSTFILTER_SER
      ${FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS}\ ./post_ensight\ --postprocessor_deprecation_warning_off\ --file=${test_directory}/xxx${IDENTIFIER}\ --output=${test_directory}/xxx${IDENTIFIER}_SER_${name_of_input_file}\ --stress=${stresstype}\ --strain=${straintype}\ --start=${startstep}
      )
  set(RUNPOSTFILTER_PAR
      ${MPIEXEC_EXECUTABLE}\ ${_mpiexec_all_args_for_testing}\ -np\ ${num_proc}\ ./post_ensight\ --postprocessor_deprecation_warning_off\ --file=${test_directory}/xxx${IDENTIFIER}\ --output=${test_directory}/xxx${IDENTIFIER}_PAR_${name_of_input_file}\ --stress=${stresstype}\ --strain=${straintype}\ --start=${startstep}
      )

  # remove file ending of input file for reference file
  get_filename_component(name_of_reference_file ${name_of_input_file} NAME_WE)

  set(RUNCOMPARISON_SER
      "post-processing-comparison ${test_directory}/xxx${IDENTIFIER}_SER_${name_of_input_file}${FIELD}*.case ${PROJECT_SOURCE_DIR}/tests/input_files/ref/${name_of_reference_file}${IDENTIFIER}${FIELD}.csv"
      )
  set(RUNCOMPARISON_PAR
      "post-processing-comparison ${test_directory}/xxx${IDENTIFIER}_PAR_${name_of_input_file}${FIELD}*.case ${PROJECT_SOURCE_DIR}/tests/input_files/ref/${name_of_reference_file}${IDENTIFIER}${FIELD}.csv"
      )

  # specify test case
  if(FOUR_C_WITH_PYTHON)
    add_test(
      NAME "${name_of_test}"
      COMMAND
        sh -c
        " ${RUNPOSTFILTER_PAR} && ${RUNPOSTFILTER_SER} && ${RUNCOMPARISON_SER} && ${RUNCOMPARISON_PAR}"
      )
  else()
    skip_test(
      ${name_of_test}
      "Skipping because FOUR_C_WITH_PYTHON is not enabled. Postprocessing tests require Python."
      )
  endif()

  require_fixture("${name_of_test}" "${name_of_input_file}-p${num_proc_base_run};test_cleanup")
  set_environment(${name_of_test})
  set_processors(${name_of_test} ${num_proc})
  set_timeout(${name_of_test})

  # Set "RUN_SERIAL TRUE" because result files can only be read by one process.
  set_run_serial(${name_of_test})
endfunction()

###------------------------------------------------------------------ Compare VTK
# Central function to define a vtk test based on an a previous input file test
#
# required parameters:
#   BASED_ON:                     Reference to previous test
#   PVD_RESULT:                   name of the .pvd file in the output directory of the previous test that references the .vtk files to compare
#   PVD_REFERENCE:                name of the .pvd file in tests/input_files that
#                                 references the .vtk files to compare against
#   TOLERANCE:                    difference the values may have
#
# optional parameters:
#   TIME_STEPS:                   Timesteps to compare
#   TIMEOUT:                      Manually defined duration for test timeout; defaults to global timeout if not specified
#   LABELS:                       Add labels to the test
#   REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                 Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                 The supported version constraint operators are: >=, <=, >, <, ==
#                                 If multiple dependencies are provided, all must be met for the test to run.
#                                 Note that the version is the _internal_ version that 4C assigns to the dependency.
function(__four_c_test_vtk)
  set(options "")
  set(oneValueArgs
      BASED_ON
      PVD_RESULT
      PVD_REFERENCE
      TOLERANCE
      TIMEOUT
      )
  set(multiValueArgs TIME_STEPS LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(
    _parsed
    BASED_ON
    PVD_RESULT
    PVD_REFERENCE
    TOLERANCE
    )

  # get test directory of base test
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  set(merged_timesteps_to_compare "")
  if(DEFINED _parsed_TIME_STEPS)
    list(JOIN _parsed_TIME_STEPS " " merged_timesteps_to_compare)
  endif()

  set(name_of_test "${_parsed_BASED_ON}-vtk-${_parsed_PVD_RESULT}")
  set(test_command
      "vtk-compare ${test_directory}/${_parsed_PVD_RESULT} ${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_PVD_REFERENCE} ${_parsed_TOLERANCE} --points_in_time ${merged_timesteps_to_compare}"
      )

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    ADDITIONAL_FIXTURE
    ${_parsed_BASED_ON}
    TOTAL_PROCS
    "1"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
endfunction()

###------------------------------------------------------------------ Compare VTK
# Central function to define a timings test based on an a previous input file test
#
# required parameters:
#   BASED_ON:                     Reference to previous test
#   TIMINGS_FILE:                 name of the timings file in the output directory of the previous test that contains the timings to compare
#   TIMERS:                       List of names of the timers to compare
#
# optional parameters:
#   MAX_TIMES:                    List of maximum times for the timers to compare (either empty or matching the length of TIMERS).
#   MAX_CALLS:                    List of maximum calls for the timers to compare (either empty or matching the length of TIMERS).
#   LABELS:                       Add labels to the test
#   REQUIRED_DEPENDENCIES:        Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                 Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                 The supported version constraint operators are: >=, <=, >, <, ==
#                                 If multiple dependencies are provided, all must be met for the test to run.
#                                 Note that the version is the _internal_ version that 4C assigns to the dependency.
function(__four_c_test_timings)
  set(options "")
  set(oneValueArgs BASED_ON TIMINGS_FILE)
  set(multiValueArgs
      TIMERS
      MAX_TIMES
      MAX_CALLS
      LABELS
      REQUIRED_DEPENDENCIES
      )
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

  assert_required_arguments(_parsed BASED_ON TIMINGS_FILE TIMERS)

  # get test directory of base test
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  set(merged_timers_to_compare "")
  if(DEFINED _parsed_TIMERS)
    list(JOIN _parsed_TIMERS " " merged_timers_to_compare)
  endif()

  set(merged_times_to_compare "")
  if(DEFINED _parsed_MAX_TIMES)
    list(JOIN _parsed_MAX_TIMES " " merged_times_to_compare)
  endif()

  set(merged_num_calls_to_compare "")
  if(DEFINED _parsed_MAX_CALLS)
    list(JOIN _parsed_MAX_CALLS " " merged_num_calls_to_compare)
  endif()

  set(name_of_test "${_parsed_BASED_ON}-timings")
  set(test_command
      "check-timings ${test_directory}/${_parsed_TIMINGS_FILE} ${merged_timers_to_compare} --expected-max-time ${merged_times_to_compare} --expected-max-num-calls ${merged_num_calls_to_compare}"
      )

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    ADDITIONAL_FIXTURE
    ${_parsed_BASED_ON}
    TOTAL_PROCS
    "1"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
endfunction()

###------------------------------------------------------------------ Restart 4C simulation and compare VTK Output
# Two-stage test case, where a simulation is restarted from same input file and
# VTK file is compared for the restarted run.
#
# Usage:
# four_c_test_restarted_vtk(
#   TESTNAME name_of_test
#   FILE <input_in_tests/input_files>
#   RESTART_STEP <step>
#   RESTARTFROM_PATHTYPE <absolute|relative|relative_from_parent|same_directory|same_directory_implicit>
#   PVD_RESULTFILENAME <subdir_with_pvd_under_sim2>
#   PVD_REFERENCEFILENAME <reference_file_under_tests/input_files>
#   TOLERANCE <tol>
#   [TIME_STEPS t1 t2 ...]
#   [LABELS l1 l2 ...]
#   [TIMEOUT <seconds>]
# )
function(four_c_test_restarted_vtk)
  set(options "")
  set(oneValueArgs
      TESTNAME
      FILE
      RESTART_STEP
      RESTARTFROM_PATHTYPE
      PVD_RESULTFILENAME
      PVD_REFERENCEFILENAME
      TOLERANCE
      TIMEOUT
      )
  set(multiValueArgs TIME_STEPS LABELS)
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  # validation of required args
  foreach(
    req IN
    ITEMS TESTNAME
          FILE
          RESTART_STEP
          RESTARTFROM_PATHTYPE
          PVD_RESULTFILENAME
          PVD_REFERENCEFILENAME
          TOLERANCE
    )
    if(NOT DEFINED _parsed_${req})
      message(FATAL_ERROR "four_c_test_restarted_vtk: missing required argument ${req}")
    endif()
  endforeach()

  # List of allowed values for RESTARTFROM_PATHTYPE
  set(allowed_restartfrom_pathtypes
      absolute
      relative
      relative_from_parent
      same_directory
      same_directory_implicit
      )
  if(NOT _parsed_RESTARTFROM_PATHTYPE IN_LIST allowed_restartfrom_pathtypes)
    list(JOIN allowed_restartfrom_pathtypes "', '" _allowed_restartfrom_pathtypes_joined)
    message(
      FATAL_ERROR
        "four_c_test_restarted_vtk: RESTARTFROM_PATHTYPE must be one of '${_allowed_restartfrom_pathtypes_joined}'"
      )
  endif()

  # verify inputs exist
  set(input_path "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_FILE}")
  if(NOT EXISTS "${input_path}")
    message(FATAL_ERROR "Input file not found: ${input_path}")
  endif()
  set(ref_path "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_PVD_REFERENCEFILENAME}")
  if(NOT EXISTS "${ref_path}")
    message(FATAL_ERROR "Reference file not found: ${ref_path}")
  endif()

  # perform test serial
  set(base_NP 1)

  # specify names and directories
  set(name_of_test
      "${_parsed_TESTNAME}-p${base_NP}-restart_step_${_parsed_RESTART_STEP}-restartfrom_pathtype_${_parsed_RESTARTFROM_PATHTYPE}"
      )
  set(root_dir ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
  set(folder_name_sim1 "simulation_1")
  set(folder_name_sim2 "simulation_2")
  set(sim1_dir ${root_dir}/${folder_name_sim1})
  set(relative_path_from_sim2_to_sim1 "../${folder_name_sim1}")
  set(sim2_dir ${root_dir}/${folder_name_sim2})

  # optional timeout (scaled)
  set(local_timeout "")
  if(DEFINED _parsed_TIMEOUT AND NOT "${_parsed_TIMEOUT}" STREQUAL "")
    math(EXPR local_timeout "${FOUR_C_TEST_TIMEOUT_SCALE} * ${_parsed_TIMEOUT}")
  endif()

  # First simulation in directory simulation_1
  set(run1_cmd
      "mkdir -p ${sim1_dir} && \
      ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} \
      $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${input_path} ${sim1_dir}/xxx"
      )

  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${run1_cmd}
    TOTAL_PROCS
    ${base_NP}
    TIMEOUT
    "${local_timeout}"
    LABELS
    "${_parsed_LABELS}"
    )

  # Restart second simulation previous directory simulation_1
  set(name_of_restart "${name_of_test}-restart")
  if(_parsed_RESTARTFROM_PATHTYPE STREQUAL "absolute") # restartfrom sim1_dir
    set(run2_cmd
        "mkdir -p ${sim2_dir} \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${input_path} ${sim2_dir}/xxx --restartfrom=${sim1_dir}/xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATHTYPE STREQUAL "relative"
         )# change into sim2_dir, then restartfrom relative_path_from_sim2_to_sim1
    set(run2_cmd
        "mkdir -p ${sim2_dir} && cd ${sim2_dir} && echo changing pwd to: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${input_path} ${sim2_dir}/xxx --restartfrom=${relative_path_from_sim2_to_sim1}/xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATHTYPE STREQUAL "relative_from_parent"
         )# change into root_dir, then restartfrom folder_name_sim1
    set(run2_cmd
        "mkdir -p ${sim2_dir} && cd ${root_dir} && echo changing pwd to: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${input_path} ${sim2_dir}/xxx --restartfrom=${folder_name_sim1}/xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATHTYPE STREQUAL "same_directory")
    # restartfrom sim1_dir (same directory), output is still written in sim2_dir for later vtk comparison
    set(run2_cmd
        "mkdir -p ${sim2_dir} && cd ${sim1_dir} && echo staying in: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${input_path} ${sim2_dir}/xxx --restartfrom=xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATHTYPE STREQUAL "same_directory_implicit")
    # restartfrom sim1_dir (same directory) and write output in same directory while not specifying restartfrom path. Note: Output is written in sim1_dir now.
    # Hence it is convenient to set sim2_dir to sim1_dir for the rest of the test such that the vtk comparison works later on.
    set(sim2_dir ${sim1_dir})
    set(run2_cmd
        "cd ${sim1_dir} && echo staying in: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${input_path} xxx --restart=${_parsed_RESTART_STEP}"
        )
  endif()
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_restart}
    TEST_COMMAND
    ${run2_cmd}
    ADDITIONAL_FIXTURE
    ${name_of_test}
    TOTAL_PROCS
    ${base_NP}
    TIMEOUT
    "${local_timeout}"
    LABELS
    "${_parsed_LABELS}"
    )
  set_run_serial(${name_of_restart})

  # compare final VTK output in directory simulation_2
  set(name_of_compare "${name_of_restart}-vtk-compare")
  if(FOUR_C_WITH_PYTHON)
    list(LENGTH _parsed_TIME_STEPS nsteps)
    if(nsteps GREATER 0)
      list(JOIN _parsed_TIME_STEPS " " steps_joined)
    else()
      set(steps_joined "")
    endif()
    set(result_dir "${sim2_dir}/${_parsed_PVD_RESULTFILENAME}")

    set(compare_cmd
        "vtk-compare ${result_dir} ${ref_path} ${_parsed_TOLERANCE} --points_in_time ${steps_joined}"
        )
    _add_test_with_options(
      NAME_OF_TEST
      ${name_of_compare}
      TEST_COMMAND
      ${compare_cmd}
      ADDITIONAL_FIXTURE
      ${name_of_restart}
      TIMEOUT
      "${local_timeout}"
      LABELS
      "${_parsed_LABELS}"
      )
    set_run_serial(${name_of_compare})
  else()
    skip_test(
      ${name_of_compare}
      "Skipping because FOUR_C_WITH_PYTHON is not enabled. VTK comparison requires Python."
      )
    require_fixture(${name_of_compare} ${name_of_restart})
  endif()
endfunction()

###------------------------------------------------------------------ Final cleanup
# remove any output files from our tests
# autogenerated core files (generated by kernel)
function(four_c_test_cleanup)
  add_test(
    NAME test_cleanup
    COMMAND
      sh -c
      "if [ -f *_CUTFAIL.pos ]; then mkdir -p ../cut-debug ; cp *_CUTFAIL.pos ../cut-debug/ ; fi ; rm -fr xxx* framework_test_output* core.* tutorials*"
    )
  set_processors(test_cleanup 1)
  set_environment(test_cleanup)
  set_tests_properties(test_cleanup PROPERTIES FIXTURES_CLEANUP test_cleanup)
endfunction()
