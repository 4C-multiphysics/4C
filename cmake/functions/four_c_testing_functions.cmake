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
  set_tests_properties(${name_of_test} PROPERTIES ENVIRONMENT "PATH=$ENV{PATH}")
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

# add test with options
function(_add_test_with_options)

  set(options "")
  set(oneValueArgs NAME_OF_TEST ADDITIONAL_FIXTURE NP TIMEOUT)
  set(multiValueArgs TEST_COMMAND LABELS)
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

  if(NOT DEFINED _parsed_NAME_OF_TEST)
    message(FATAL_ERROR "Name of test is a necessary input argument!")
  endif()

  if(NOT DEFINED _parsed_TEST_COMMAND)
    message(FATAL_ERROR "Test command is a necessary input argument!")
  endif()

  if(NOT DEFINED _parsed_ADDITIONAL_FIXTURE)
    set(_parsed_ADDITIONAL_FIXTURE "")
  endif()

  if(NOT DEFINED _parsed_NP)
    set(_parsed_NP 1)
  endif()

  if(NOT DEFINED _parsed_TIMEOUT)
    set(_parsed_TIMEOUT "")
  endif()

  if(NOT DEFINED _parsed_LABELS)
    set(_parsed_LABELS "")
  endif()

  add_test(NAME ${_parsed_NAME_OF_TEST} COMMAND bash -c "${_parsed_TEST_COMMAND}")

  require_fixture(${_parsed_NAME_OF_TEST} "${_parsed_ADDITIONAL_FIXTURE};test_cleanup")
  set_processors(${_parsed_NAME_OF_TEST} ${_parsed_NP})
  define_setup_fixture(${_parsed_NAME_OF_TEST} ${_parsed_NAME_OF_TEST})
  set_timeout(${_parsed_NAME_OF_TEST} ${_parsed_TIMEOUT})

  if(NOT ${_parsed_LABELS} STREQUAL "")
    set_label(${_parsed_NAME_OF_TEST} ${_parsed_LABELS})
  endif()

endfunction()

###------------------------------------------------------------------ 4C Test
# Run simulation with input file
# Usage in tests/lists_of_tests.cmake:
#            "four_c_test(<input_file> optional: NP <> RESTART_STEP <> TIMEOUT <> OMP_THREADS <> LABEL <>
#                                                CSV_YAML_COMPARISON_RESULT_FILE <> CSV_YAML_COMPARISON_REFERENCE_FILE <>
#                                                CSV_YAML_COMPARISON_TOL_R <> CSV_YAML_COMPARISON_TOL_A <>)"

# TEST_FILE:              must equal the name of an input file in directory tests/input_files
#                         If two files are provided the second input file is restarted based on the results of the first input file.

# optional:
# NP:                             Number of processors the test should use. Fallback to 1 if not specified.
#                                 For two input files two NP's are required.
# RESTART_STEP:                   Number of restart step; not defined indicates no restart
# TIMEOUT:                        Manually defined duration for test timeout; defaults to global timeout if not specified
# OMP_THREADS:                    Number of OpenMP threads per processor the test should use; defaults to deactivated
# LABELS:                         Add labels to the test
# CSV_YAML_COMPARISON_RESULT_FILE:     Arbitrary .csv result files to be compared (see `utilites/diff_with_tolerance.py`)
# CSV_YAML_COMPARISON_REFERENCE_FILE:  Reference files to compare with
# CSV_YAML_COMPARISON_TOL_R:           Relative tolerances for comparison
# CSV_YAML_COMPARISON_TOL_A:           Absolute tolerances for comparison
# REQUIRED_DEPENDENCIES:          Any required external dependencies. The test will be skipped if the dependencies are not met.

function(four_c_test)

  set(options "")
  set(oneValueArgs RESTART_STEP TIMEOUT OMP_THREADS)
  set(multiValueArgs
      TEST_FILE
      NP
      LABELS
      CSV_YAML_COMPARISON_RESULT_FILE
      CSV_YAML_COMPARISON_REFERENCE_FILE
      CSV_YAML_COMPARISON_TOL_R
      CSV_YAML_COMPARISON_TOL_A
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

  if(NOT DEFINED _parsed_TEST_FILE)
    message(FATAL_ERROR "Test file is required for test!")
  endif()

  if(NOT DEFINED _parsed_NP)
    set(_parsed_NP 1)
  endif()

  if(NOT DEFINED _parsed_RESTART_STEP)
    set(_parsed_RESTART_STEP "")
  endif()

  if(NOT DEFINED _parsed_TIMEOUT)
    set(_parsed_TIMEOUT "")
  endif()

  if(NOT DEFINED _parsed_OMP_THREADS)
    set(_parsed_OMP_THREADS 0)
  endif()

  if(NOT DEFINED _parsed_LABELS)
    set(_parsed_LABELS "")
  endif()

  if(NOT DEFINED _parsed_CSV_YAML_COMPARISON_RESULT_FILE)
    set(_parsed_CSV_YAML_COMPARISON_RESULT_FILE "")
  endif()

  if(NOT DEFINED _parsed_CSV_YAML_COMPARISON_REFERENCE_FILE)
    set(_parsed_CSV_YAML_COMPARISON_REFERENCE_FILE "")
  endif()

  if(NOT DEFINED _parsed_CSV_YAML_COMPARISON_TOL_R)
    set(_parsed_CSV_YAML_COMPARISON_TOL_R "")
  endif()

  if(NOT DEFINED _parsed_CSV_YAML_COMPARISON_TOL_A)
    set(_parsed_CSV_YAML_COMPARISON_TOL_A "")
  endif()

  list(LENGTH _parsed_TEST_FILE num_TEST_FILE)
  list(LENGTH _parsed_NP num_NP)
  if(num_TEST_FILE GREATER 2 OR NOT num_NP EQUAL num_TEST_FILE)
    message(
      FATAL_ERROR
        "You provided more than two test files or the provided number of processors do not match your provided test files!"
      )
  endif()

  # Assert that NP <= 3
  foreach(_np IN LISTS _parsed_NP)
    if(_np GREATER 3)
      message(FATAL_ERROR "Number of processors must be less than or equal to 3!")
    endif()
  endforeach()

  # check if source files exist
  set(source_file "")
  foreach(string IN LISTS _parsed_TEST_FILE)
    set(file_name "${PROJECT_SOURCE_DIR}/tests/input_files/${string}")
    if(NOT EXISTS ${file_name})
      message(FATAL_ERROR "Test source file ${file_name} does not exist!")
    endif()
    list(APPEND source_file ${file_name})
  endforeach()

  # check that same number of files and tolerances are provided for arbitrary .csv comparison
  list(LENGTH _parsed_CSV_YAML_COMPARISON_RESULT_FILE num_CSV_YAML_COMPARISON_RESULT_FILE)
  list(LENGTH _parsed_CSV_YAML_COMPARISON_REFERENCE_FILE num_CSV_YAML_COMPARISON_REFERENCE_FILE)
  list(LENGTH _parsed_CSV_YAML_COMPARISON_TOL_R num_CSV_YAML_COMPARISON_TOL_R)
  list(LENGTH _parsed_CSV_YAML_COMPARISON_TOL_A num_CSV_YAML_COMPARISON_TOL_A)
  if(NOT num_CSV_YAML_COMPARISON_RESULT_FILE EQUAL num_CSV_YAML_COMPARISON_REFERENCE_FILE
     OR NOT num_CSV_YAML_COMPARISON_RESULT_FILE EQUAL num_CSV_YAML_COMPARISON_TOL_R
     OR NOT num_CSV_YAML_COMPARISON_RESULT_FILE EQUAL num_CSV_YAML_COMPARISON_TOL_A
     )
    message(
      FATAL_ERROR
        "You must provide the same number of files and tolerances for arbitrary .csv comparison!"
      )
  endif()

  # check if .csv reference files exists
  set(CSV_YAML_COMPARISON_REFERENCE_FILE "")
  foreach(string IN LISTS _parsed_CSV_YAML_COMPARISON_REFERENCE_FILE)
    set(file_name "${PROJECT_SOURCE_DIR}/tests/input_files/${string}")
    if(NOT EXISTS ${file_name})
      message(
        FATAL_ERROR
          "Reference file ${file_name} for arbitrary .csv result comparison does not exist!"
        )
    endif()
    list(APPEND CSV_YAML_COMPARISON_REFERENCE_FILE ${file_name})
  endforeach()

  if(DEFINED _parsed_REQUIRED_DEPENDENCIES)
    foreach(dep IN LISTS _parsed_REQUIRED_DEPENDENCIES)
      four_c_sanitize_package_name(${dep} dep_sanitized)
      # Check that we even know this dependency
      if(NOT DEFINED "FOUR_C_WITH_${dep_sanitized}")
        message(FATAL_ERROR "Unknown dependency ${dep}")
      endif()
      if(NOT FOUR_C_WITH_${dep_sanitized})
        string(
          APPEND skip_message "Skipping because FOUR_C_WITH_${dep_sanitized} is not enabled.\n"
          )
      endif()
    endforeach()
    if(NOT "${skip_message}" STREQUAL "")
      # Note that this only adds _one_ skipped test, even though multiple dependent tests
      # might have been created by this function.
      skip_test(${_parsed_TEST_FILE}-p${_parsed_NP} "${skip_message}")
      message(VERBOSE "Skipping test ${_parsed_TEST_FILE} because dependencies are not met.")
      return()
    endif()
  endif()

  # set base test name and directory
  if(num_TEST_FILE EQUAL 1)
    set(base_test_file ${source_file})
    set(base_NP ${_parsed_NP})
    set(name_of_test ${_parsed_TEST_FILE}-p${_parsed_NP})
    set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
  elseif(num_TEST_FILE EQUAL 2)
    list(GET source_file 0 base_test_file)
    list(GET source_file 1 restart_test_file)
    list(GET _parsed_TEST_FILE 0 base_test)
    list(GET _parsed_TEST_FILE 1 restart_test)
    list(GET _parsed_NP 0 base_NP)
    list(GET _parsed_NP 1 restart_NP)
    set(name_of_test
        ${base_test}-p${base_NP}_for_${restart_test}-p${restart_NP}-restart_step_${_parsed_RESTART_STEP}
        )
    set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
  endif()

  set(test_command
      "mkdir -p ${test_directory} \
                && ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${base_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${base_test_file} ${test_directory}/xxx"
      )

  # Optional timeout
  if(NOT "${_parsed_TIMEOUT}" STREQUAL "")
    # scale testtimeout with the global test timeout scale
    math(EXPR _parsed_TIMEOUT "${FOUR_C_TEST_TIMEOUT_SCALE} * ${_parsed_TIMEOUT}")
  endif()

  # Optional OpenMP threads per processor
  set(total_procs ${base_NP})
  if(${_parsed_OMP_THREADS})
    set(name_of_test ${name_of_test}-OMP${_parsed_OMP_THREADS})
    set(test_command
        "export OMP_NUM_THREADS=${_parsed_OMP_THREADS} && ${test_command} && unset OMP_NUM_THREADS"
        )
    math(EXPR total_procs "${_parsed_NP}*${_parsed_OMP_THREADS}")
  endif()

  _add_test_with_options(
    NAME_OF_TEST
    ${name_of_test}
    TEST_COMMAND
    ${test_command}
    NP
    ${total_procs}
    TIMEOUT
    "${_parsed_TIMEOUT}"
    LABELS
    "${_parsed_LABELS}"
    )

  # set additional fixture for restart
  set(additional_fixture "${name_of_test}")

  # restart option
  if(NOT _parsed_RESTART_STEP STREQUAL "" AND num_TEST_FILE EQUAL 1)
    # restart with same input file
    set(name_of_test "${name_of_test}-restart_from_same_input")
    set(test_command "${test_command} restart=${_parsed_RESTART_STEP}")
    _add_test_with_options(
      NAME_OF_TEST
      ${name_of_test}
      TEST_COMMAND
      ${test_command}
      ADDITIONAL_FIXTURE
      ${additional_fixture}
      NP
      ${total_procs}
      TIMEOUT
      "${_parsed_TIMEOUT}"
      LABELS
      "${_parsed_LABELS}"
      )

    # update additional fixture for possible following post_ensight or csv comparison
    set(additional_fixture "${name_of_test}")

  elseif(NOT _parsed_RESTART_STEP STREQUAL "" AND num_TEST_FILE EQUAL 2)
    # restart with different input file
    set(name_of_test
        ${restart_test}-p${restart_NP}_from_${base_test}-p${base_NP}-restart_step_${_parsed_RESTART_STEP}
        )
    set(restart_test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_test})
    set(test_command
        "mkdir -p ${restart_test_directory} && ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np ${restart_NP} $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> ${restart_test_file} ${restart_test_directory}/xxx restartfrom=${test_directory}/xxx restart=${_parsed_RESTART_STEP}"
        )

    # Optional OpenMP threads per processor
    set(total_procs ${restart_NP})
    if(${_parsed_OMP_THREADS})
      set(name_of_test ${name_of_test}-OMP${_parsed_OMP_THREADS})
      set(test_command
          "export OMP_NUM_THREADS=${_parsed_OMP_THREADS} && ${test_command} && unset OMP_NUM_THREADS"
          )
      math(EXPR total_procs "${_parsed_NP}*${_parsed_OMP_THREADS}")
    endif()

    _add_test_with_options(
      NAME_OF_TEST
      ${name_of_test}
      TEST_COMMAND
      ${test_command}
      ADDITIONAL_FIXTURE
      ${additional_fixture}
      NP
      ${total_procs}
      TIMEOUT
      "${_parsed_TIMEOUT}"
      LABELS
      "${_parsed_LABELS}"
      )
    set_run_serial(${name_of_test})

    # update additional fixture for possible following post_ensight or csv comparison
    set(additional_fixture "${name_of_test}")

  endif()

  # csv comparison
  if(NOT _parsed_CSV_YAML_COMPARISON_RESULT_FILE STREQUAL "")

    # loop over all csv comparisons
    foreach(
      result_file
      reference_file
      tol_r
      tol_a
      IN
      ZIP_LISTS
      _parsed_CSV_YAML_COMPARISON_RESULT_FILE
      _parsed_CSV_YAML_COMPARISON_REFERENCE_FILE
      _parsed_CSV_YAML_COMPARISON_TOL_R
      _parsed_CSV_YAML_COMPARISON_TOL_A
      )
      set(name_of_csv_comparison_test "${name_of_test}-csv_comparison-${result_file}")
      if(FOUR_C_WITH_PYTHON)
        set(csv_comparison_command
            "${FOUR_C_PYTHON_VENV_BUILD}/bin/python ${PROJECT_SOURCE_DIR}/utilities/diff_with_tolerance.py ${test_directory}/${result_file} ${PROJECT_SOURCE_DIR}/tests/input_files/${reference_file} ${tol_r} ${tol_a}"
            )
        _add_test_with_options(
          NAME_OF_TEST
          ${name_of_csv_comparison_test}
          TEST_COMMAND
          ${csv_comparison_command}
          ADDITIONAL_FIXTURE
          ${additional_fixture}
          LABELS
          "${_parsed_LABELS}"
          )
      else()
        skip_test(
          ${name_of_csv_comparison_test} "Skipping because FOUR_C_WITH_PYTHON is not enabled."
          )
      endif()
    endforeach()
  endif()

endfunction()

###------------------------------------------------------------------ Nested Parallelism
# Usage in tests/lists_of_tests.cmake: "four_c_test_nested_parallelism(<name_of_input_file_1> <name_of_input_file_2> <restart_step>)"
# <name_of_input_file_1>: must equal the name of an input file in directory tests/input_files for the first test; This test will be executed using 1 process.
# <name_of_input_file_2>: must equal the name of an input file in directory tests/input_files for the second test; This test will be executed using 2 processes.
# <restart_step>: number of restart step; <""> indicates no restart
function(four_c_test_nested_parallelism name_of_input_file_1 name_of_input_file_2 restart_step)
  set(test_directory ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_input_file_1})

  add_test(
    NAME ${name_of_input_file_1}-nestedPar
    COMMAND
      bash -c
      "mkdir -p ${test_directory} &&  ${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np 3 $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> -ngroup=2 -glayout=1,2 -nptype=separateInputFiles ${PROJECT_SOURCE_DIR}/tests/input_files/${name_of_input_file_1} ${test_directory}/xxx ${PROJECT_SOURCE_DIR}/tests/input_files/${name_of_input_file_2} ${test_directory}/xxxAdditional"
    )

  require_fixture(${name_of_input_file_1}-nestedPar test_cleanup)
  set_processors(${name_of_input_file_1}-nestedPar 3)
  define_setup_fixture(${name_of_input_file_1}-nestedPar ${name_of_input_file_1}-nestedPar-p3)
  set_timeout(${name_of_input_file_1}-nestedPar)

  if(${restart_step})
    add_test(
      NAME ${name_of_input_file_1}-nestedPar-restart
      COMMAND
        bash -c
        "${MPIEXEC_EXECUTABLE} ${_mpiexec_all_args_for_testing} -np 3 $<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}> -ngroup=2 -glayout=1,2 -nptype=separateInputFiles ${PROJECT_SOURCE_DIR}/tests/input_files/${name_of_input_file_1} ${test_directory}/xxx restart=${restart_step} ${PROJECT_SOURCE_DIR}/tests/input_files/${name_of_input_file_2} ${test_directory}/xxxAdditional restart=${restart_step}"
      )

    require_fixture(
      ${name_of_input_file_1}-nestedPar-restart "${name_of_input_file_1}-nestedPar-p3;test_cleanup"
      )
    set_processors(${name_of_input_file_1}-nestedPar-restart 3)
  endif()
endfunction()

###------------------------------------------------------------------ Tutorial Tests
# Testing a tutorial example
#
# Usage in tests/lists_of_tests.cmake:
#
#  four_c_test_tutorial(PREFIX <prefix> NP <NP> [COPY_FILES <file1> <file2> ...])"
#
#  PREFIX: must equal the name of a .4C.yaml and a .e file in directory tests/tutorials
#  NP: number of MPI ranks for this test
#  COPY_FILES: copy any additional files to the test directory
function(four_c_test_tutorial)
  set(oneValueArgs PREFIX NP)
  set(multiValueArgs COPY_FILES)
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

  set(name_of_input_file ${_parsed_PREFIX})
  set(num_proc ${_parsed_NP})
  set(name_of_test ${name_of_input_file}-p${num_proc}-fw)
  set(test_directory tutorials/${name_of_input_file})

  list(
    APPEND
    _run_copy_files
    "cp ${PROJECT_SOURCE_DIR}/tests/tutorials/${name_of_input_file}.4C.yaml ${test_directory}/xxx.4C.yaml"
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

  add_test(
    NAME ${name_of_test}
    COMMAND
      bash -c
      "mkdir -p ${PROJECT_BINARY_DIR}/${test_directory} && ${_run_copy_files} && ${_run_4C}"
    )

  require_fixture(${name_of_test} test_cleanup)
  set_environment(${name_of_test})
  set_fail_expression(${name_of_test})
  set_processors(${name_of_test} ${num_proc})
  set_timeout(${name_of_test})
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
      ${FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS}\ ./post_ensight\ --file=${test_directory}/xxx${IDENTIFIER}\ --output=${test_directory}/xxx${IDENTIFIER}_SER_${name_of_input_file}\ --stress=${stresstype}\ --strain=${straintype}\ --start=${startstep}
      )
  set(RUNPOSTFILTER_PAR
      ${MPIEXEC_EXECUTABLE}\ ${_mpiexec_all_args_for_testing}\ -np\ ${num_proc}\ ./post_ensight\ --file=${test_directory}/xxx${IDENTIFIER}\ --output=${test_directory}/xxx${IDENTIFIER}_PAR_${name_of_input_file}\ --stress=${stresstype}\ --strain=${straintype}\ --start=${startstep}
      )

  # remove file ending of input file for reference file
  get_filename_component(name_of_reference_file ${name_of_input_file} NAME_WE)

  # specify test case
  if(FOUR_C_WITH_PYTHON)
    add_test(
      NAME "${name_of_test}"
      COMMAND
        sh -c
        " ${RUNPOSTFILTER_PAR} && ${RUNPOSTFILTER_SER} && ${FOUR_C_PVPYTHON} ${PROJECT_SOURCE_DIR}/tests/post_processing_test/comparison.py ${test_directory}/xxx${IDENTIFIER}_PAR_${name_of_input_file}${FIELD}*.case ${test_directory}/xxx${IDENTIFIER}_SER_${name_of_input_file}${FIELD}*.case ${PROJECT_SOURCE_DIR}/tests/input_files/${name_of_reference_file}${IDENTIFIER}${FIELD}.csv ${test_directory}"
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
# Compare XML formatted .vtk result data set referenced by .pvd files to corresponding reference files
# CAUTION: This tests bases on results of a previous simulation/test
# Implementation can be found in '/tests/output_test/vtk_compare.py'
# Usage in tests/lists_of_tests.cmake: "four_c_test_vtk(<name_of_input_file> <num_proc> <filetag> <pvd_referencefilename> <tolerance> <optional: time_steps>)"
# <name_of_test>: name of this test
# <name_of_input_file>: must equal the name of an input file from a previous test
# <num_proc_base_run>: number of processors of precursor base run
# <pvd_referencefilename>: file to compare with
# <tolerance>: difference the values may have
# <optional: time_steps>: time steps when to compare
function(
  four_c_test_vtk
  name_of_test
  name_of_input_file
  num_proc_base_run
  pvd_resultfilename
  pvd_referencefilename
  tolerance
  )
  set(test_directory
      ${PROJECT_BINARY_DIR}/framework_test_output/${name_of_input_file}-p${num_proc_base_run}/${pvd_resultfilename}
      )

  # this test takes a list of times as extra arguments to check results at those timesteps
  # if no extra arguments are given test checks every timestep
  set(extra_macro_args ${ARGN})

  # Did we get any optional args?
  list(LENGTH extra_macro_args num_extra_args)

  if(${num_extra_args} GREATER 0)
    list(GET extra_macro_args 0 optional_arg)
  endif()

  if(FOUR_C_WITH_PYTHON)
    # add test to testing framework
    add_test(
      NAME "${name_of_test}-p${num_proc_base_run}"
      COMMAND
        ${FOUR_C_PYTHON_VENV_BUILD}/bin/python
        ${PROJECT_SOURCE_DIR}/tests/output_test/vtk_compare.py ${test_directory}
        ${PROJECT_SOURCE_DIR}/tests/input_files/${pvd_referencefilename} ${tolerance}
        ${num_extra_args} ${extra_macro_args}
      )
  else()
    skip_test(
      "${name_of_test}-p${num_proc_base_run}"
      "Skipping because FOUR_C_WITH_PYTHON is not enabled. Postprocessing tests require Python."
      )
  endif()

  require_fixture(
    ${name_of_test}-p${num_proc_base_run}
    "${name_of_input_file}-p${num_proc_base_run};${name_of_input_file}-p${num_proc_base_run}-restart;test_cleanup"
    )
  set_processors(${name_of_test}-p${num_proc_base_run} 1)
  set_timeout(${name_of_test}-p${num_proc_base_run})
endfunction()

###------------------------------------------------------------------ Final cleanup
# remove any output files from our tests
# autogenerated core files (generated by kernel)
add_test(
  NAME test_cleanup
  COMMAND
    sh -c
    "if [ -f *_CUTFAIL.pos ]; then mkdir -p ../cut-debug ; cp *_CUTFAIL.pos ../cut-debug/ ; fi ; rm -fr xxx* framework_test_output* core.* tutorials*"
  )
set_processors(test_cleanup 1)
set_tests_properties(test_cleanup PROPERTIES FIXTURES_CLEANUP test_cleanup)
