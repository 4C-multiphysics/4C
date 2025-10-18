# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Build system for 4C and support tools.

#
### General Setup

cmake_minimum_required(VERSION 3.30)

# try to prevent modification of source directory
# note: some files may still be written before CMake can abort and need to be removed manually
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  message(
    FATAL_ERROR
      "In-source build not allowed. "
      "Please create a new directory, preferably next to the source directory, and run CMake from there. "
      "You may want to remove CMakeCache.txt and CMakeFiles/ which were created in the source directory."
    )
endif()

project(4C CXX C)

# Print CMake version to screen
message(STATUS "Using CMake ${CMAKE_VERSION}")

# Enforce the C++ standard we are using and turn off compiler-specific extensions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# We do not use C++ modules (yet). Turn off scanning to avoid issues with clang-tidy.
# If you want to add module support, this problem needs to be revisited. Our hope
# is that CMake and/or clang-tidy will be updated to handle modules better.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Ensure cmake setup the correct runtime path when installing
# see here for more information: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
include(GNUInstallDirs)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Read the version from the VERSION file. Instead of encoding the version in the CMakeLists.txt,
# the file allows easy access to the version from other scripts.
# Following https://calver.org/, the version string has to look like this:  YYYY.MINOR.PATCH[-MODIFIER]
# where PATCH is optional and implicitly 0 if not present.
file(STRINGS "${PROJECT_SOURCE_DIR}/VERSION" FOUR_C_VERSION_FULL LIMIT_COUNT 1)
string(
  REGEX MATCH
        "([0-9][0-9][0-9][0-9])\.([0-9]+)\.([0-9]+)(-([a-z0-9]+))?"
        _dummy
        ${FOUR_C_VERSION_FULL}
  )
if(NOT _dummy)
  message(
    FATAL_ERROR
      "The version string ${FOUR_C_VERSION_FULL} does not match the expected format YYYY.MINOR.PATCH[-MODIFIER]."
    )
endif()
set(FOUR_C_VERSION_MAJOR ${CMAKE_MATCH_1})
set(FOUR_C_VERSION_MINOR ${CMAKE_MATCH_2})
set(FOUR_C_VERSION_PATCH ${CMAKE_MATCH_3})

message(STATUS "Configuring 4C version ${FOUR_C_VERSION_FULL}")

# The input version is used to determine the compatibility of input files.
# Syntax: major.minor.patch
file(
  STRINGS "${PROJECT_SOURCE_DIR}/input_version/INPUT_VERSION" FOUR_C_INPUT_VERSION_FULL
  LIMIT_COUNT 1
  )
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _dummy ${FOUR_C_INPUT_VERSION_FULL})
if(NOT _dummy)
  message(
    FATAL_ERROR
      "The input version string ${FOUR_C_INPUT_VERSION_FULL} does not match the expected format MAJOR.MINOR.PATCH"
    )
endif()
set(FOUR_C_INPUT_VERSION_MAJOR ${CMAKE_MATCH_1})
set(FOUR_C_INPUT_VERSION_MINOR ${CMAKE_MATCH_2})
set(FOUR_C_INPUT_VERSION_PATCH ${CMAKE_MATCH_3})
message(STATUS "The 4C input version is ${FOUR_C_INPUT_VERSION_FULL}")

# Assert a valid build configuration
set(_allowed_build_types "DEBUG" "RELEASE" "RELWITHDEBINFO")
if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Choose a build type. Options are: ${_allowed_build_types}.")
else()
  string(TOUPPER ${CMAKE_BUILD_TYPE} FOUR_C_BUILD_TYPE_UPPER)
  if(NOT ${FOUR_C_BUILD_TYPE_UPPER} IN_LIST _allowed_build_types)
    message(
      FATAL_ERROR "Invalid build type ${CMAKE_BUILD_TYPE}. Options are: ${_allowed_build_types}."
      )
  endif()
  message(STATUS "Configuring for build type ${CMAKE_BUILD_TYPE}")
endif()

# specify where our own cmake modules are located
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# include all external (cmake) macros that we use
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)

include(FetchContent)

# Include our own functions and macros
file(GLOB _function_files CONFIGURE_DEPENDS "cmake/functions/*.cmake")
foreach(_file ${_function_files})
  message(VERBOSE "Include ${_file}")
  include(${_file})
endforeach()

# Perform basic checks
file(GLOB _checks CONFIGURE_DEPENDS "cmake/checks/*.cmake")
foreach(_file ${_checks})
  message(VERBOSE "Include ${_file}")
  include(${_file})
endforeach()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

### process global options
include(cmake/setup_global_options.cmake)

###------------------------------------------------------------------ External Libraries

# Collect all external dependencies in a helper target
add_library(four_c_all_enabled_external_dependencies INTERFACE)

four_c_configure_dependency(HDF5 REQUIRED)
four_c_configure_dependency(MPI REQUIRED)
four_c_configure_dependency(Qhull REQUIRED)
four_c_configure_dependency(Trilinos REQUIRED)
four_c_configure_dependency(VTK DEFAULT OFF)
four_c_configure_dependency(deal.II DEFAULT OFF)
four_c_configure_dependency(Boost REQUIRED)
four_c_configure_dependency(ArborX DEFAULT OFF)
four_c_configure_dependency(FFTW REQUIRED)
four_c_configure_dependency(CLN REQUIRED)
four_c_configure_dependency(MIRCO DEFAULT OFF)
four_c_configure_dependency(Backtrace DEFAULT OFF)
four_c_configure_dependency(ryml REQUIRED)
four_c_configure_dependency(magic_enum REQUIRED)
four_c_configure_dependency(ZLIB REQUIRED)
four_c_configure_dependency(Python DEFAULT ON) # not required but useful for most developers
four_c_configure_dependency(pybind11 DEFAULT OFF)

### end of configuration and feature detection

# Enable testing right away since we encounter src and tests interspersed in modules
enable_testing()

# NOTE: --mca orte_tmpdir_base [...] can be removed for openmpi >= 4.1.1 according to
# https://github.com/open-mpi/ompi/issues/8510#issuecomment-1329297350
four_c_process_cache_variable(
  FOUR_C_MPIEXEC_ARGS_FOR_TESTING
  TYPE
  STRING
  DESCRIPTION
  "Arguments to pass to mpiexec for testing."
  DEFAULT
  "--bind-to none --use-hwthread-cpus --mca orte_tmpdir_base ${PROJECT_BINARY_DIR}/tmp"
  )
# We might need to add more arguments. Initialize with the user input.
set(_mpiexec_all_args_for_testing ${FOUR_C_MPIEXEC_ARGS_FOR_TESTING})

if(FOUR_C_ENABLE_ADDRESS_SANITIZER)
  # Do not detect leaks, we only care about UB inducing memory issues
  string(APPEND _mpiexec_all_args_for_testing " -x LSAN_OPTIONS=detect_leaks=0")
  set(FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS "LSAN_OPTIONS=detect_leaks=0")
endif()

set(FOUR_C_EXECUTABLE_NAME 4C)
set(FULL_TARGETS post_processor post_monitor create_rtd cut_test)

add_custom_target(full DEPENDS ${FOUR_C_EXECUTABLE_NAME} ${FULL_TARGETS})

include(cmake/setup_tests.cmake)

# Create one large library: every module adds itself to this target
set(FOUR_C_LIBRARY_NAME lib4C)
add_library(${FOUR_C_LIBRARY_NAME})
# Remove the CMake generated prefix "lib" because we already added it manually, to have a distinct target name.
set_target_properties(${FOUR_C_LIBRARY_NAME} PROPERTIES PREFIX "")
target_link_libraries(${FOUR_C_LIBRARY_NAME} PRIVATE four_c_private_compile_interface)
if(NOT FOUR_C_BUILD_SHARED_LIBS)
  install(
    TARGETS four_c_private_compile_interface
    EXPORT 4CTargets
    FILE_SET HEADERS
    )
endif()

# Read in the source directory which defines all modules
add_subdirectory(src)
add_subdirectory(apps)

# Process documentation
add_subdirectory(doc EXCLUDE_FROM_ALL)

# Tests
add_subdirectory(unittests)
add_subdirectory(tests/benchmark_tests)
add_subdirectory(tests/cut_test)
include(tests/list_of_tests.cmake)

# Installation
include(cmake/setup_install.cmake)
