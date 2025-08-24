# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

if(FOUR_C_BUILD_TYPE_UPPER STREQUAL "DEBUG")
  set(_default_timeout_scale 4)
else()
  set(_default_timeout_scale 1)
endif()
four_c_process_cache_variable(
  FOUR_C_TEST_TIMEOUT_SCALE
  TYPE
  STRING
  DESCRIPTION
  "Scale timeout of tests by this factor."
  DEFAULT
  ${_default_timeout_scale}
  )

math(EXPR FOUR_C_TEST_GLOBAL_TIMEOUT "120*${FOUR_C_TEST_TIMEOUT_SCALE}")
message(STATUS "The scaled global test timeout is ${FOUR_C_TEST_GLOBAL_TIMEOUT} s.")

four_c_process_cache_variable(
  FOUR_C_PVPYTHON
  TYPE
  FILEPATH
  DESCRIPTION
  "Path to the pvpython executable used for post-processing tests"
  DEFAULT
  "pvpython-not-set"
  )

# Fetch GoogleTest and setup the unit tests if option is enabled
four_c_process_global_option(
  FOUR_C_WITH_GOOGLETEST
  DESCRIPTION
  "Use GoogleTest for unit testing"
  DEFAULT
  ON
  )
if(FOUR_C_WITH_GOOGLETEST)
  # Define a convenience target for all unit tests and add it to 'full'
  # All unit test executables should add themselves as a dependency to 'unittests'
  add_custom_target(unittests)
  add_dependencies(full unittests)

  if(TARGET gtest)
    message(
      FATAL_ERROR "A target <gtest> has already been included by another library."
                  "This is not supported."
      )
  endif()

  include(FetchContent)
  fetchcontent_declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
    )
  fetchcontent_makeavailable(googletest)

  math(EXPR UNITTEST_TIMEOUT "10*${FOUR_C_TEST_TIMEOUT_SCALE}")
  message(STATUS "The scaled unit test timeout is ${UNITTEST_TIMEOUT} s.")

else()
  message(STATUS "Unit tests with GoogleTest: disabled")
endif()

# Fetch Google Benchmark and setup benchmark tests if option is enabled
four_c_process_global_option(
  FOUR_C_WITH_GOOGLE_BENCHMARK
  DESCRIPTION
  "Use Google Benchmark for micro benchmark tests"
  DEFAULT
  OFF
  )
if(FOUR_C_WITH_GOOGLE_BENCHMARK)
  # Define a convenience target for all benchmark tests and add it to 'full'
  # All benchmark test executables should add themselves as a dependency to 'benchmarktests'
  # disable tests from google benchmark
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_custom_target(benchmarktests)
  add_dependencies(full benchmarktests)

  if(TARGET benchmark_main)
    message(
      FATAL_ERROR "A target <benchmark_main> has already been included by another library."
                  "This is not supported."
      )
  endif()

  include(FetchContent)
  fetchcontent_declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG afa23b7699c17f1e26c88cbf95257b20d78d6247 # v1.9.2
    )
  fetchcontent_makeavailable(googlebenchmark)

else()
  message(STATUS "Benchmark tests with Google Benchmark: disabled")
endif()

# setup test for installation
set(FOURC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/cmake/4C)
configure_file(
  ${PROJECT_SOURCE_DIR}/tests/install_test/main.cpp.in
  ${PROJECT_BINARY_DIR}/tests/install_test/main.cpp
  @ONLY
  )
configure_file(
  ${PROJECT_SOURCE_DIR}/tests/install_test/CMakeLists.txt.in
  ${PROJECT_BINARY_DIR}/tests/install_test/CMakeLists.txt
  @ONLY
  )
configure_file(
  ${PROJECT_SOURCE_DIR}/tests/install_test/test_install.sh.in
  ${PROJECT_BINARY_DIR}/tests/install_test/test_install.sh
  @ONLY
  )
