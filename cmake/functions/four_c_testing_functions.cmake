# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

###------------------------------------------------------------------ Helper Functions

# define this test (name_of_test) as a "setup fixture" (named name_of_fixture). Other tests may
# add a dependency on such a setup fixture through the require_fixtures() function. CTest will
# then ensure that a) required fixtures are included if necessary and b) tests are executed in
# the correct order.
#
# In the 4C test suite we need such dependencies between a test e.g. for an initial simulation
# and a test for a restart, or between an initial simulation and a post-processing test.
function(define_setup_fixture name_of_test name_of_fixture)
  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_SETUP ${name_of_fixture})
endfunction()

# adds required tests (name_of_required_tests) to this test (name_of_test). The required
# tests must be defined through define_setup_fixture() as "setup fixtures". For more details on why
# these functions are needed, have a look at the documentation of define_setup_fixture().
function(require_fixtures name_of_test name_of_required_tests)
  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_REQUIRED "${name_of_required_tests}")
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

# Verify whether a (unique) global property has already been registered; otherwise, register it with a given value
function(_register_unique_global_property global_property value error_message)
  get_property(
    _already_registered GLOBAL
    PROPERTY "${global_property}"
    SET
    )
  if(_already_registered)
    message(FATAL_ERROR "${error_message}")
  else()
    set_property(GLOBAL PROPERTY "${global_property}" "${value}")
  endif()
endfunction()

# Internal helper that adds a test to the 4C test suite. Do not call this directly.
function(_add_test_with_options)
  set(options "")
  set(oneValueArgs
      NAME_OF_TEST
      TOTAL_PROCS
      TIMEOUT
      SKIP
      INPUT_FILE
      OUTPUT_DIR
      )
  set(multiValueArgs
      TEST_COMMAND
      REQUIRED_DEPENDENCIES
      LABELS
      EXCLUDE_PLATFORM
      ADDITIONAL_FIXTURES
      CLEANUP_FIXTURES
      )
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

  if(NOT DEFINED _parsed_ADDITIONAL_FIXTURES)
    set(_parsed_ADDITIONAL_FIXTURES "")
  endif()

  if(NOT DEFINED _parsed_TOTAL_PROCS)
    set(_parsed_TOTAL_PROCS 1)
  endif()

  if(NOT DEFINED _parsed_SKIP)
    set(_parsed_SKIP FALSE)
  endif()

  if(NOT DEFINED _parsed_TIMEOUT)
    set(_parsed_TIMEOUT "")
  endif()

  if(NOT DEFINED _parsed_LABELS)
    set(_parsed_LABELS "")
  endif()

  # ensure that the cleanup fixture is added
  list(APPEND _parsed_CLEANUP_FIXTURES "test_cleanup")

  # add all required dependencies from the additional fixtures, if specified
  foreach(fixture IN LISTS _parsed_ADDITIONAL_FIXTURES)
    append_required_dependencies_from(${fixture} _parsed_REQUIRED_DEPENDENCIES)
  endforeach()

  # check if all required dependencies are present
  check_required_dependencies(skip_message "${_parsed_REQUIRED_DEPENDENCIES}")

  # check if the test is marked to be skipped
  if(_parsed_SKIP)
    set(skip_message "The test ${_parsed_NAME_OF_TEST} is marked to be skipped")
  endif()

  if(NOT skip_message STREQUAL "")
    # The dummy test needs to report a arbitrary error code that ctest interprets as "skipped".
    set(dummy_command "echo \"${skip_message}\"; exit 42")
    # Add a dummy test that just prints the skip message instead of the real test
    add_test(NAME ${_parsed_NAME_OF_TEST} COMMAND bash -c "${dummy_command}")
    set_tests_properties(${_parsed_NAME_OF_TEST} PROPERTIES SKIP_RETURN_CODE 42)
    message(VERBOSE "Skipping test ${_parsed_NAME_OF_TEST}: ${skip_message}")

    # Note: Cleanup fixtures are not required for skipped tests
    require_fixtures(${_parsed_NAME_OF_TEST} "${_parsed_ADDITIONAL_FIXTURES}")
  else()
    # Add the real test
    add_test(NAME ${_parsed_NAME_OF_TEST} COMMAND bash -c "${_parsed_TEST_COMMAND}")

    require_fixtures(
      ${_parsed_NAME_OF_TEST} "${_parsed_ADDITIONAL_FIXTURES};${_parsed_CLEANUP_FIXTURES}"
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
#   EXCLUDE_PLATFORM:             Mark to not run the test on specific platform. By default we enable the test, but if EXCLUDE_PLATFORM is defined,
#                                 e.g., Linux|Darwin, then the test is ignored if the platform is matched.
function(four_c_test)
  set(options "")
  set(oneValueArgs
      TEST_FILE
      NP
      TIMEOUT
      OMP_THREADS
      RETURN_AS
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES EXCLUDE_PLATFORM)
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

  # check if the platform is matched
  set(skip_var FALSE)
  if(CMAKE_SYSTEM_NAME IN_LIST _parsed_EXCLUDE_PLATFORM)
    set(skip_var TRUE)
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
    SKIP
    "${skip_var}"
    )
endfunction()

##
# Define a restart test that restarts from a previous test.
#
# required parameters:
#   BASED_ON:                name of the base test that created the restart files
#   RESTART_STEP:            number of the restart step to restart from or last_possible
#   SAME_FILE or TEST_FILE:  either SAME_FILE to indicate that the restart should be done from the same input file
#                            as the base test, or TEST_FILE to indicate that a different input file should be used for the restart.
#                            Note that in the case of SAME_FILE, the restart output is also saved within the same directory as the base test;
#                            in the case of TEST_FILE, a new test directory is created (except when RESTARTFROM_PATH_TYPE is same_directory_implicit!).
#
# optional parameters:
#   RESTART_OUTPUT_PREFIX    prefix for the restart output files within the output directory. Defaults to "xxx".
#                            Cannot be specified if the --restartfrom type is "same_directory_implicit".
#                            Note that we try to keep the restart output prefixes unambiguous - multiple SAME_FILE restarts require
#                            to define specific restart output prefixes. For TEST_FILE, it's fine if the name of the test differs
#                            (this sets the output directory, see implementation below), even if the restart output prefix is the same for all such restart tests.
#                            To keep restart output prefixes completely unambiguous, we don't allow for more than one restart test
#                            using --restartfrom=same_directory_implicit (also holds for multiple SAME_FILE / TEST_FILE restart tests using this --restartfrom #                            option).
#   RESTARTFROM_PATH_TYPE     <absolute|relative|relative_from_parent|same_directory|same_directory_implicit>. Defaults to absolute if not specified.
#   NP:                      number of processors the test should use. Fallback to 1 if not specified.
#   TIMEOUT:                 manually defined duration for test timeout; defaults to global timeout if not specified
#   OMP_THREADS:             number of OpenMP threads per processor the test should use; defaults to no OpenMP if not specified
#   LABELS:                  Add labels to the test
#   REQUIRED_DEPENDENCIES:   Any required external dependencies. The test will be skipped if the dependencies are not met.
#   RETURN_AS:               A variable name that allows to add further dependent tests based on this test.
#   EXCLUDE_PLATFORM:        Mark to not run the restart test on specific platform. By default we enable the test, but if EXCLUDE_PLATFORM is defined,
#                            e.g., Linux|Darwin, then the test is ignored if the platform is matched.
function(__four_c_test_restart)
  set(options SAME_FILE)
  set(oneValueArgs
      BASED_ON
      TEST_FILE
      RESTARTFROM_PATH_TYPE
      RESTART_OUTPUT_PREFIX
      NP
      RESTART_STEP
      TIMEOUT
      OMP_THREADS
      RETURN_AS
      ASSERT_RESTART_STEP
      )
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES EXCLUDE_PLATFORM)
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )

  # List of allowed values for RESTARTFROM_PATH_TYPE
  set(allowed_restartfrom_pathtypes
      absolute
      relative
      relative_from_parent
      same_directory
      same_directory_implicit
      )
  if(NOT DEFINED _parsed_RESTARTFROM_PATH_TYPE)
    set(_parsed_RESTARTFROM_PATH_TYPE "absolute")
  endif()
  if(DEFINED _parsed_RESTART_OUTPUT_PREFIX
     AND _parsed_RESTARTFROM_PATH_TYPE STREQUAL "same_directory_implicit"
     )
    message(
      FATAL_ERROR
        "You cannot set the restart output prefix if the --restartfrom type is same_directory_implicit"
      )
  endif()
  if(NOT DEFINED _parsed_RESTART_OUTPUT_PREFIX)
    set(_parsed_RESTART_OUTPUT_PREFIX "xxx")
  endif()
  if(NOT _parsed_RESTARTFROM_PATH_TYPE IN_LIST allowed_restartfrom_pathtypes)
    list(JOIN allowed_restartfrom_pathtypes "', '" _allowed_restartfrom_pathtypes_joined)
    message(
      FATAL_ERROR
        "__four_c_test_restart: RESTARTFROM_PATH_TYPE must be one of '${_allowed_restartfrom_pathtypes_joined}'"
      )
  endif()

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

  if(_parsed_SAME_FILE AND DEFINED _parsed_TEST_FILE)
    message(FATAL_ERROR "You cannot specify both SAME_FILE and TEST_FILE")
  endif()
  if(NOT _parsed_SAME_FILE AND NOT DEFINED _parsed_TEST_FILE)
    message(FATAL_ERROR "You must specify either SAME_FILE or TEST_FILE")
  endif()

  # In case we reuse the same file as the base test, get the input file from there, and also use the same output directory.
  # We differentiate in the following between base_directory, which contains the output files from the BASED_ON simulation, and test_directory, where
  # this restart test will run.
  if(_parsed_SAME_FILE)
    set(name_of_test
        "${_parsed_BASED_ON}-restart_${_parsed_RESTART_STEP}-p${_parsed_NP}-${_parsed_RESTART_OUTPUT_PREFIX}"
        )
    # ensure that this restart output prefix has not already been registered; since this test will run in the same simulation directory as BASED_ON,
    # we specifically check the restart output prefix (we want to avoid repeating restart output prefixes in the same simulation directory)
    _register_unique_global_property(
      "_restart_output_prefix_${_parsed_RESTART_OUTPUT_PREFIX}_for_${_parsed_BASED_ON}"
      1
      "You have to define another restart output prefix, since ${_parsed_RESTART_OUTPUT_PREFIX} already exists for ${_parsed_BASED_ON}"
      )

    get_test_property(${_parsed_BASED_ON} _internal_INPUT_FILE test_file_full_path)
    get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)
    set(base_directory ${test_directory})
  else()
    # Restart from a different testfile
    set(name_of_test
        "${_parsed_BASED_ON}-restart_${_parsed_RESTART_STEP}-with_${_parsed_TEST_FILE}-p${_parsed_NP}-${_parsed_RESTART_OUTPUT_PREFIX}"
        )
    # ensure that this restart output prefix has not already been registered
    if(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "same_directory_implicit")
      # For implicit same-directory restarts: check if restart output files using the default prefix (xxx) have already been written;
      # should not be the case if this test is to be run
      _register_unique_global_property(
        "_restart_output_prefix_${_parsed_RESTART_OUTPUT_PREFIX}_for_${_parsed_BASED_ON}"
        1
        "You try to run a same_directory_implicit restart using TEST_FILE when there is already a SAME_FILE restart with default restart output prefix for ${_parsed_BASED_ON}! This is not supported since it leads to ambiguous restart output prefixes xxx-<count> which may be subject to race conditions!"
        )
    else()
      # Else: check whether this test exists (the restart simulations run in directories specified with the test name
      # - hence, it's fine if they have the same restart output prefix as long as they do not run in the same directory,
      # i.e., as long as they don't have the same test name)
      check_test_exists(_restart_test_file_exists ${name_of_test})
      if(_restart_test_file_exists)
        message(
          FATAL_ERROR
            "You have to define another restart output prefix, since ${name_of_test} already exists for ${_parsed_BASED_ON}."
          )
      endif()
    endif()
    set(test_file_full_path "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_TEST_FILE}")
    set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
    get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR base_directory)

    if(NOT EXISTS ${test_file_full_path})
      message(FATAL_ERROR "Test source file ${test_file_full_path} does not exist")
    endif()
  endif()

  # set root directory and required relative paths
  set(root_directory "${PROJECT_BINARY_DIR}/framework_test_output")
  file(RELATIVE_PATH relative_path_from_test_to_base_dir "${test_directory}" "${base_directory}")
  if(relative_path_from_test_to_base_dir STREQUAL "")
    set(relative_path_from_test_to_base_dir ".")
  endif()
  file(RELATIVE_PATH relative_path_from_root_to_base_dir "${root_directory}" "${base_directory}")
  if(relative_path_from_root_to_base_dir STREQUAL "")
    message(
      FATAL_ERROR
        "The root directory ${root_directory} of the tests is the same as the base directory ${base_directory} of ${_parsed_BASED_ON}"
      )
  endif()

  if(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "absolute")
    set(test_command
        "mkdir -p ${test_directory} \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${test_file_full_path} ${test_directory}/${_parsed_RESTART_OUTPUT_PREFIX} --restartfrom=${base_directory}/xxx --restart=${_parsed_RESTART_STEP}"
        )

  elseif(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "relative"
         )# change into the test directory, then restart from the relative path
    set(test_command
        "mkdir -p ${test_directory} && cd ${test_directory} && echo changing pwd to: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${test_file_full_path} ${test_directory}/${_parsed_RESTART_OUTPUT_PREFIX} --restartfrom=${relative_path_from_test_to_base_dir}/xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "relative_from_parent"
         )# change into the root directory, then restart from there
    set(test_command
        "mkdir -p ${test_directory} && cd ${root_directory} && echo changing pwd to: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${test_file_full_path} ${test_directory}/${_parsed_RESTART_OUTPUT_PREFIX} --restartfrom=${relative_path_from_root_to_base_dir}/xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "same_directory")
    # restart from base directory, output is still written in test directory
    set(test_command
        "mkdir -p ${test_directory} && cd ${base_directory} && echo staying in: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${test_file_full_path} ${test_directory}/${_parsed_RESTART_OUTPUT_PREFIX} --restartfrom=xxx --restart=${_parsed_RESTART_STEP}"
        )
  elseif(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "same_directory_implicit")
    # restart from base directory and write output in the same directory while not specifying restartfrom path.
    # Note: Output is written in base directory now. Hence it is convenient to set the test directory to base directory for the rest of the test.
    set(test_directory ${base_directory})
    set(test_command
        "cd ${base_directory} && echo staying in: && pwd \
      && ${extra_env}${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${_parsed_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> \
      ${test_file_full_path} ${_parsed_RESTART_OUTPUT_PREFIX} --restart=${_parsed_RESTART_STEP}"
        )
  endif()

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

  # skip the restart test if the main test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

  # check if the platform is matched
  if(CMAKE_SYSTEM_NAME IN_LIST _parsed_EXCLUDE_PLATFORM)
    set(skip_var TRUE)
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
    ADDITIONAL_FIXTURES
    ${_parsed_BASED_ON}
    TOTAL_PROCS
    "${total_procs}"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    SKIP
    "${skip_var}"
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
    if(_parsed_RESTARTFROM_PATH_TYPE STREQUAL "same_directory_implicit"
       OR (_parsed_SAME_FILE AND _parsed_RESTART_OUTPUT_PREFIX STREQUAL "xxx")
       )
      set(control_file "${test_directory}/xxx-1.control")
    else()
      set(control_file "${test_directory}/${_parsed_RESTART_OUTPUT_PREFIX}.control")
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
      ADDITIONAL_FIXTURES
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
#   EXCLUDE_PLATFORM:      mark to not run the test on specific platform.
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
  set(multiValueArgs LABELS REQUIRED_DEPENDENCIES EXCLUDE_PLATFORM)
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

  # skip the test if the main test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

  # check if the platform is matched
  if(CMAKE_SYSTEM_NAME IN_LIST _parsed_EXCLUDE_PLATFORM)
    set(skip_var TRUE)
  endif()

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_csv_comparison_test}
    TEST_COMMAND
    ${csv_comparison_command}
    ADDITIONAL_FIXTURES
    ${_parsed_BASED_ON}
    SKIP
    "${skip_var}"
    LABELS
    "${_parsed_LABELS}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    SKIP
    "${skip_var}"
    )
endfunction()

##
# Define a comparison test that compares a csv or yaml result file created by one test against
# the result file created by another test
#
# required parameters:
#   BASED_ON:              name of the first base test that created the first result file
#   COMPARE_TO:            name of the second base test that created the second result file
#   RESULT_FILE:           name of the result file created by BASED_ON
#   COMPARE_FILE:          name of the result file created by COMPARE_TO
#   TOL_R:                 relative tolerance for comparison
#   TOL_A:                 absolute tolerance for comparison
#
# optional parameters:
#   LABELS:                add labels to the test
#   REQUIRED_DEPENDENCIES: any required external dependencies. The test will be skipped if the dependencies are not met.
#
function(__four_c_test_add_csv_yaml_comparison_between_tests)
  set(options "")
  set(oneValueArgs
      BASED_ON
      COMPARE_TO
      RESULT_FILE
      COMPARE_FILE
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

  if(DEFINED _parsed_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "There are unparsed arguments: ${_parsed_UNPARSED_ARGUMENTS}!")
  endif()

  assert_required_arguments(
    _parsed
    BASED_ON
    COMPARE_TO
    RESULT_FILE
    COMPARE_FILE
    TOL_R
    TOL_A
    )

  set(name_of_csv_comparison_test
      "${_parsed_BASED_ON}-csv_comparison-${_parsed_COMPARE_TO}-${_parsed_RESULT_FILE}"
      )
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  # skip the test if the "base" test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

  get_test_property(${_parsed_COMPARE_TO} _internal_OUTPUT_DIR compare_directory)

  set(csv_comparison_command
      "diff-with-tolerance ${test_directory}/${_parsed_RESULT_FILE} ${compare_directory}/${_parsed_COMPARE_FILE} ${_parsed_TOL_R} ${_parsed_TOL_A}"
      )

  append_required_dependencies_from(${_parsed_BASED_ON} _parsed_REQUIRED_DEPENDENCIES)
  append_required_dependencies_from(${_parsed_COMPARE_TO} _parsed_REQUIRED_DEPENDENCIES)
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_csv_comparison_test}
    TEST_COMMAND
    ${csv_comparison_command}
    SKIP
    "${skip_var}"
    LABELS
    "${_parsed_LABELS}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
  set_tests_properties(
    ${name_of_csv_comparison_test}
    PROPERTIES FIXTURES_REQUIRED "${_parsed_BASED_ON};${_parsed_COMPARE_TO};test_cleanup"
    )
endfunction()

##
# Define a comparison test that checks the header row of a CSV result file.
#
# required parameters:
#   BASED_ON:              name of the base test that created the result file to compare against
#   RESULT_FILE:           name of the result file created by the base test
#   HEADERS:               expected CSV header entries
#
# optional parameters:
#   LABELS:                add labels to the test
#   REQUIRED_DEPENDENCIES: any required external dependencies. The test will be skipped if the dependencies are not met.
#
function(four_c_test_add_csv_header_check)
  set(options "")
  set(oneValueArgs BASED_ON RESULT_FILE)
  set(multiValueArgs HEADERS LABELS REQUIRED_DEPENDENCIES)
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

  assert_required_arguments(_parsed BASED_ON RESULT_FILE HEADERS)

  set(name_of_csv_header_test "${_parsed_BASED_ON}-csv_header_check-${_parsed_RESULT_FILE}")
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  # skip the test if the "base" test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

  set(csv_header_check_command
      "${FOUR_C_PYTHON_VENV_BUILD}/bin/check-csv-headers ${test_directory}/${_parsed_RESULT_FILE}"
      )
  foreach(header IN LISTS _parsed_HEADERS)
    string(APPEND csv_header_check_command " \"${header}\"")
  endforeach()

  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_csv_header_test}
    TEST_COMMAND
    ${csv_header_check_command}
    ADDITIONAL_FIXTURES
    ${_parsed_BASED_ON}
    SKIP
    "${skip_var}"
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
      ADDITIONAL_FIXTURES
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
# EXCLUDE_PLATFORM:              Mark to not run the test on specific platform. By default we enable the test, but if EXCLUDE_PLATFORM is defined,
#                                e.g., Linux|Darwin, then the test is ignored if the platform is matched.
function(four_c_test_tutorial)
  set(oneValueArgs TEST_FILE NP TIMEOUT)
  set(multiValueArgs COPY_FILES REQUIRED_DEPENDENCIES EXCLUDE_PLATFORM)
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

  # check if the test is marked to be skipped
  if(skip_message STREQUAL "")
    if(CMAKE_SYSTEM_NAME IN_LIST _parsed_EXCLUDE_PLATFORM)
      set(skip_message "The test ${name_of_test} is marked to be skipped")
    endif()
  endif()

  if(NOT skip_message STREQUAL "")
    # The dummy test needs to report a arbitrary error code that ctest interprets as "skipped".
    set(dummy_command "echo \"${skip_message}\"; exit 42")
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

  require_fixtures(${name_of_test} test_cleanup)
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

  require_fixtures(${name_of_test} test_cleanup)
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

  require_fixtures("${name_of_test}" "${name_of_input_file}-p${num_proc_base_run};test_cleanup")
  set_environment(${name_of_test})
  set_processors(${name_of_test} ${num_proc})
  set_timeout(${name_of_test})

  # Set "RUN_SERIAL TRUE" because result files can only be read by one process.
  set_run_serial(${name_of_test})
endfunction()

###------------------------------------------------------------------ Compare VTK
# Central function to define a vtk test for a previous input file test: either against a reference pvd file, or against another input file test (e.g., corresponding restart test)
#
# required parameters:
#   BASED_ON:                      Reference to previous test for which the comparison is run
#   PVD_RESULT_FILE:               Name of the .pvd file in the output directory of the previous test that references the .vtk files to compare
#   PVD_REFERENCE_FILE:            Name of the .pvd reference file: if REFERENCE_TEST is specified, this file is in the output directory of that reference test; otherwise, the path of this reference file is in tests/input_files
#   TOLERANCE_DATA:                Absolute tolerance for comparing the vtk data
#   TOLERANCE_TIME:                Absolute tolerance for comparing time instants which may be shifted due to round-off, default: 0.0
#
# optional parameters:
#   REFERENCE_TEST:                Optional: Name of the fixture to be used as reference
#   TIME_STEPS:                    Timesteps to compare
#   TIMEOUT:                       Manually defined duration for test timeout; defaults to global timeout if not specified
#   LABELS:                        Add labels to the test
#   REQUIRED_DEPENDENCIES:         Any required external dependencies. The test will be skipped if the dependencies are not met.
#                                  Either a dependency, e.g. "Trilinos", or a dependency with a version constraint, e.g. "Trilinos>=2025.2".
#                                  The supported version constraint operators are: >=, <=, >, <, ==
#                                  If multiple dependencies are provided, all must be met for the test to run.
#                                  Note that the version is the _internal_ version that 4C assigns to the dependency.
function(__four_c_test_vtk)
  set(options "")
  set(oneValueArgs
      BASED_ON
      PVD_RESULT_FILE
      PVD_REFERENCE_FILE
      REFERENCE_TEST
      TOLERANCE_DATA
      TOLERANCE_TIME
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
    PVD_RESULT_FILE
    PVD_REFERENCE_FILE
    TOLERANCE_DATA
    )

  # set default comparison tolerance for time instants
  if(NOT DEFINED _parsed_TOLERANCE_TIME)
    set(_parsed_TOLERANCE_TIME 0.0)
  endif()

  # get test directory of base test
  get_test_property(${_parsed_BASED_ON} _internal_OUTPUT_DIR test_directory)

  set(merged_timesteps_to_compare "")
  if(DEFINED _parsed_TIME_STEPS)
    list(JOIN _parsed_TIME_STEPS " " merged_timesteps_to_compare)
  endif()

  # determine comparison target
  set(additional_fixtures "${_parsed_BASED_ON}")
  if(DEFINED _parsed_REFERENCE_TEST)
    get_test_property(${_parsed_REFERENCE_TEST} _internal_OUTPUT_DIR reference_test_directory)
    set(reference_pvd "${reference_test_directory}/${_parsed_PVD_REFERENCE_FILE}")
    list(APPEND additional_fixtures "${_parsed_REFERENCE_TEST}")
    set(name_of_test
        "${_parsed_BASED_ON}-vtk-${_parsed_PVD_RESULT_FILE}-compare-with-${_parsed_REFERENCE_TEST}-${_parsed_PVD_REFERENCE_FILE}"
        )
  else()
    set(reference_pvd "${PROJECT_SOURCE_DIR}/tests/input_files/${_parsed_PVD_REFERENCE_FILE}")
    set(name_of_test
        "${_parsed_BASED_ON}-vtk-${_parsed_PVD_RESULT_FILE}-compare-with-existing-pvd-${_parsed_PVD_REFERENCE_FILE}"
        )
  endif()

  set(test_command
      "vtk-compare ${test_directory}/${_parsed_PVD_RESULT_FILE} ${reference_pvd} --tolerance_data ${_parsed_TOLERANCE_DATA} --tolerance_time ${_parsed_TOLERANCE_TIME} --points_in_time ${merged_timesteps_to_compare}"
      )

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")

  # skip the test if the main test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    ADDITIONAL_FIXTURES
    ${additional_fixtures}
    TOTAL_PROCS
    "1"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    SKIP
    "${skip_var}"
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
#   MIN_CALLS:                    List of minimum calls for the timers to compare (either empty or matching the length of TIMERS).
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
      MIN_CALLS
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

  # skip the test if the "base" test is skipped
  get_test_property(${_parsed_BASED_ON} SKIP_RETURN_CODE skip_code_base_var)
  if(skip_code_base_var STREQUAL "NOTFOUND")
    set(skip_var FALSE)
  else()
    set(skip_var TRUE)
  endif()

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

  set(merged_min_num_calls_to_compare "")
  if(DEFINED _parsed_MIN_CALLS)
    list(JOIN _parsed_MIN_CALLS " " merged_min_num_calls_to_compare)
  endif()

  set(name_of_test "${_parsed_BASED_ON}-timings")
  set(test_command
      "check-timings ${test_directory}/${_parsed_TIMINGS_FILE} ${merged_timers_to_compare} --expected-max-time ${merged_times_to_compare} --expected-max-num-calls ${merged_num_calls_to_compare} --expected-min-num-calls ${merged_min_num_calls_to_compare}"
      )

  # Ensure that Python is listed as required dependency
  list(APPEND _parsed_REQUIRED_DEPENDENCIES "Python")

  # Add test
  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    ADDITIONAL_FIXTURES
    ${_parsed_BASED_ON}
    TOTAL_PROCS
    "1"
    TIMEOUT
    "${_parsed_TIMEOUT}"
    SKIP
    "${skip_var}"
    LABELS
    "${_parsed_LABELS}"
    OUTPUT_DIR
    "${test_directory}"
    REQUIRED_DEPENDENCIES
    "${_parsed_REQUIRED_DEPENDENCIES}"
    )
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
