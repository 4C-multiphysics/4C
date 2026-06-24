# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(_set_up_benchmark_test_target _module_under_test _target)
  message(VERBOSE "Setting up benchmark test target ${_target}")

  add_executable(
    ${_target}
    ${PROJECT_SOURCE_DIR}/tests/benchmark_tests/4C_benchmark_tests_main.cpp ${_parsed_SOURCE}
    )

  # Store benchmark test executables directly inside the tests/ directory
  set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

  # Do not try to build benchmark tests as unity files.
  set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)

  # All libraries are linked as PRIVATE since a benchmark test executable cannot be used as a dependency itself.

  # Common dependencies for benchmark tests
  target_link_libraries(${_target} PRIVATE four_c_private_compile_interface)

  target_link_libraries(${_target} PRIVATE ${_module_under_test}_module)

  target_link_libraries(${_target} PRIVATE benchmark::benchmark)

  target_link_libraries(${_target} PRIVATE unittests_common)

  if(FOUR_C_CLANGCUDA)
    set_clangcuda_mode(${_target} CLANGCUDA_MODE_DEVICE)
  endif()

  # configure the respective input test with the respective mesh
  if(FOUR_C_ENABLE_FULL_BENCHMARK_TESTS)
    set(benchmark_test_additional_options "")
  else()
    set(benchmark_test_additional_options "--benchmark_dry_run")
  endif()

  set(test_output_directory ${PROJECT_BINARY_DIR}/framework_test_output/benchmark_tests)

  add_test(
    NAME ${_target}
    COMMAND
      bash -c
      "mkdir -p ${test_output_directory} \
       && ${FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS}\ $<TARGET_FILE:${_target}>\ ${benchmark_test_additional_options} --benchmark_out=${test_output_directory}/${_target}-benchmark.json --benchmark_out_format=json"
    )
  set_tests_properties(${_target} PROPERTIES TIMEOUT ${BENCHMARK_TEST_TIMEOUT})
  set_tests_properties(${_target} PROPERTIES PROCESSORS 1)
  set_tests_properties(${_target} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
  set_tests_properties(${_target} PROPERTIES LABELS "benchmark_tests")

  require_fixture(${_target} "collect_benchmark_test_results")

  add_dependencies(benchmarktests ${_target})
endfunction()

##
# Pickup all the source files in the current directory and add them to a benchmark test target.
# Recursively add all subdirectories that contain CMakeLists.txt files. A module under test
# may be supplied via MODULE. If no module name is supplied, the name of the parent module is used.
##
function(four_c_auto_define_benchmark_tests)
  if(NOT FOUR_C_WITH_GOOGLE_BENCHMARK)
    return()
  endif()

  set(options "")
  set(oneValueArgs MODULE)
  set(multiValueArgs "")
  cmake_parse_arguments(
    _parsed
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
    )
  if(DEFINED _parsed_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "There are unparsed arguments: ${_parsed_UNPARSED_ARGUMENTS}")
  endif()

  if(_parsed_MODULE)
    set(_module_under_test ${_parsed_MODULE})
  else()
    if("${FOUR_C_CURRENTLY_DEFINED_PARENT_MODULE}" STREQUAL "")
      message(
        FATAL_ERROR
          "No parent module is set. Either give the module this test belongs to or call this functions inside a module."
        )
    endif()

    set(_module_under_test "${FOUR_C_CURRENTLY_DEFINED_PARENT_MODULE}")
  endif()

  if(NOT TARGET "${_module_under_test}_objs")
    message(
      FATAL_ERROR
        "Tried to add tests for a module named '${_module_under_test}' which is not a known module name."
      )
  endif()

  set(_test_name_base "benchmarktests_${_module_under_test}")

  file(GLOB_RECURSE _sources CONFIGURE_DEPENDS *.cpp)

  # Treat every source file separately: determine test requirements and add to or create new test target.
  foreach(_source ${_sources})
    set(_current_test_name ${_test_name_base})

    if(NOT TARGET ${_current_test_name})
      _set_up_benchmark_test_target(${_module_under_test} ${_current_test_name})
    endif()

    target_sources(${_current_test_name} PRIVATE ${_source})
  endforeach()

  # Recursively add all subdirectories that contain CMakeLists.txt files.
  # N.B. We need to directly glob for CMakeLists.txt files here to ensure
  # the glob reruns when a new CMakeLists.txt is added.
  file(
    GLOB children
    RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*/CMakeLists.txt
    )
  foreach(child ${children})
    get_filename_component(_subdir ${child} DIRECTORY)
    add_subdirectory(${_subdir})
  endforeach()

  # Simulate a "return" by setting a variable at the call site
  set(AUTO_DEFINED_TEST_NAME
      ${_test_name_base}
      PARENT_SCOPE
      )
endfunction()

###------------------------------------------------------------------ Benchmark test
# A central function to collect the results of the benchmark tests and save them in a json file.
#
# required parameters:
#   TARGET_FILE:                  Path to the json file where the results should be saved.
#
# optional parameters:
#   ALLOW_EMPTY:                  If this label is set, the collection will not fail if no benchmark test results are found.
function(four_c_collect_benchmark_test_results)
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

  set(name_of_test benchmark_test_results_collection)

  if("${_parsed_TARGET_FILE}" STREQUAL "")
    message(FATAL_ERROR "The TARGET_FILE argument must not be empty.")
  endif()

  set(allow_empty_flag "")
  if(_parsed_ALLOW_EMPTY)
    set(allow_empty_flag "--allow-empty")
  endif()

  set(test_command
      "collect-benchmark-test-results ${PROJECT_BINARY_DIR}/framework_test_output/benchmark_tests ${_parsed_TARGET_FILE} ${allow_empty_flag}"
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

  set_tests_properties(${name_of_test} PROPERTIES FIXTURES_CLEANUP collect_benchmark_test_results)
endfunction()
