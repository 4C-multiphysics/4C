# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(_set_up_unit_test_target _module_under_test _target)
  set(options "")
  set(oneValueArgs NP THREADS)
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

  if(NOT DEFINED _parsed_NP)
    set(_parsed_NP 1)
  endif()

  if(NOT DEFINED _parsed_THREADS)
    set(_parsed_THREADS 1)
  endif()

  set(assert_mpi_file ${CMAKE_CURRENT_BINARY_DIR}/assert_mpi_${_target}_${_parsed_NP}.cpp)
  set(FOUR_C_ADD_GOOGLE_TEST_EXECUTABLE_NP ${_parsed_NP})
  configure_file(${PROJECT_SOURCE_DIR}/unittests/assert_mpi.cpp.in ${assert_mpi_file})

  message(VERBOSE "Setting up unit test target ${_target}")

  add_executable(
    ${_target}
    ${PROJECT_SOURCE_DIR}/unittests/4C_gtest_main_mpi_test.cpp ${assert_mpi_file} ${_parsed_SOURCE}
    )
  # Store unit test executables directly inside the tests/ directory
  set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

  # Do not try to build tests as unity files.
  set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)

  # All libraries are linked as PRIVATE since a unit test executable cannot be used as a dependency itself.

  # Common dependencies for unit tests
  target_link_libraries(${_target} PRIVATE four_c_private_compile_interface)
  target_link_libraries(${_target} PRIVATE gtest gmock)
  target_link_libraries(${_target} PRIVATE unittests_common)

  target_link_libraries(${_target} PRIVATE ${_module_under_test}_unit_test_deps)

  # the first process will write a unit test report
  separate_arguments(
    MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST UNIX_COMMAND ${MPIEXEC_EXTRA_OPTS_FOR_TESTING}
    )

  set(mpi_arguments
      ${MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST}
      -np
      1
      $<TARGET_FILE:${_target}>
      --gtest_output=xml:unittest_reports/${_target}_report.xml
      )
  # if there is more than one process, spawn the remaining ones without a report
  if(_parsed_NP GREATER "1")
    math(EXPR remaining_procs "${_parsed_NP}-1")
    list(
      APPEND
      mpi_arguments
      :
      ${MPIEXEC_EXTRA_OPTS_FOR_TESTING_LIST}
      -np
      ${remaining_procs}
      $<TARGET_FILE:${_target}>
      )
  endif()

  # Calculate the total number of processors required
  math(EXPR TOTAL_NUM_PROCESSORS "${_parsed_NP}*${_parsed_THREADS}")

  add_test(NAME ${_target} COMMAND ${MPIEXEC_EXECUTABLE} ${mpi_arguments})
  set_tests_properties(${_target} PROPERTIES TIMEOUT ${UNITTEST_TIMEOUT} LABELS minimal)
  set_tests_properties(${_target} PROPERTIES PROCESSORS ${TOTAL_NUM_PROCESSORS})
  set_tests_properties(${_target} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${_parsed_THREADS}")

  add_dependencies(unittests ${_target})
endfunction()

##
# Pickup all the source files in the current directory and add them to a unit test target.
# Recursively add all subdirectories that contain CMakeLists.txt files. A module under test
# may be supplied via MODULE. If no module name is supplied, the name of the parent module is used.
#
# A source file is analyzed for certain suffixes, to determine how the test should be run. The supported
# suffixes are:
#
# - .npX.: Run the test with X MPI processes.
# - .threadsX.: Run the test with X OpenMP threads.
##
function(four_c_auto_define_tests)
  if(NOT FOUR_C_WITH_GOOGLETEST)
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

  set(_test_name_base "unittests_${_module_under_test}")

  file(GLOB_RECURSE _sources CONFIGURE_DEPENDS *.cpp)

  # Treat every source file separately: determine test requirements and add to or create new test target.
  foreach(_source ${_sources})
    set(_current_test_name ${_test_name_base})

    get_filename_component(_source_name ${_source} NAME)

    string(REGEX MATCH ".*\\.np([0-9]+)\\..*" _np_match ${_source_name})
    if(_np_match)
      set(_np ${CMAKE_MATCH_1})
      set(_current_test_name ${_current_test_name}.np${_np})
    else()
      set(_np 1)
    endif()

    string(REGEX MATCH ".*\\.threads([0-9]+)\\..*" _threads_match ${_source_name})
    if(_threads_match)
      set(_threads ${CMAKE_MATCH_1})
      set(_current_test_name ${_current_test_name}.threads${_threads})
    else()
      set(_threads 1)
    endif()

    if(NOT TARGET ${_current_test_name})
      _set_up_unit_test_target(
        ${_module_under_test}
        ${_current_test_name}
        NP
        ${_np}
        THREADS
        ${_threads}
        )

      # Use the same support file directory for all flavors
      set(FOUR_C_TEST_SUPPORT_FILE_DIR
          "${CMAKE_BINARY_DIR}/tests/support_files/${_test_name_base}/"
          )
      target_compile_definitions(
        ${_current_test_name}
        PRIVATE -DFOUR_C_TEST_SUPPORT_FILE_DIR="${FOUR_C_TEST_SUPPORT_FILE_DIR}"
        )
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

##
# Copy support files to the test directory. The support files are copied to a directory that is shared by all tests
# that use the given test base name.
##
function(four_c_add_support_files_to_test _test_name_base)
  if(NOT FOUR_C_WITH_GOOGLETEST)
    return()
  endif()

  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SUPPORT_FILES)
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

  set(FOUR_C_TEST_SUPPORT_FILE_DIR "${CMAKE_BINARY_DIR}/tests/support_files/${_test_name_base}/")

  foreach(_support_file ${_parsed_SUPPORT_FILES})
    # Get file name relative to current list file
    cmake_path(RELATIVE_PATH _support_file)
    message(
      DEBUG
      "Copying support file ${_support_file} to ${FOUR_C_TEST_SUPPORT_FILE_DIR}/${_support_file}"
      )

    configure_file(${_support_file} ${FOUR_C_TEST_SUPPORT_FILE_DIR}/${_support_file})
  endforeach()

endfunction()
