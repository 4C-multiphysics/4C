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

# The version number.
set(FOUR_C_VERSION_MAJOR 1)
set(FOUR_C_VERSION_MINOR 0)

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

four_c_configure_dependency(HDF5 DEFAULT ON)
four_c_configure_dependency(MPI DEFAULT ON)
four_c_configure_dependency(Qhull DEFAULT ON)
four_c_configure_dependency(Trilinos DEFAULT ON)
four_c_configure_dependency(Boost DEFAULT ON)
four_c_configure_dependency(ArborX DEFAULT OFF)
four_c_configure_dependency(FFTW DEFAULT ON)
four_c_configure_dependency(CLN DEFAULT ON)
four_c_configure_dependency(MIRCO DEFAULT OFF)
four_c_configure_dependency(Backtrace DEFAULT OFF)
four_c_configure_dependency(yaml-cpp DEFAULT ON)

# Generate the macro definition for all dependencies automatically
get_property(FOUR_C_FLAGS_EXTERNAL_DEPENDENCIES GLOBAL PROPERTY FOUR_C_FLAGS_EXTERNAL_DEPENDENCIES)
foreach(_dependency ${FOUR_C_FLAGS_EXTERNAL_DEPENDENCIES})
  if(${_dependency})
    string(APPEND FOUR_C_DEFINES_FOR_EXTERNAL_DEPENDENCIES "#define ${_dependency}\n")
  else()
    string(APPEND FOUR_C_DEFINES_FOR_EXTERNAL_DEPENDENCIES "/* #undef ${_dependency} */\n")
  endif()
endforeach()

### end of configuration and feature detection

# Enable testing right away since we encounter src and tests interspersed in modules
enable_testing()

# NOTE: --mca orte_tmpdir_base [...] can be removed for openmpi >= 4.1.1 according to
# https://github.com/open-mpi/ompi/issues/8510#issuecomment-1329297350
set(MPIEXEC_EXTRA_OPTS_FOR_TESTING
    "--bind-to none --use-hwthread-cpus --mca orte_tmpdir_base ${PROJECT_BINARY_DIR}/tmp"
    )

if(FOUR_C_ENABLE_ADDRESS_SANITIZER)
  # Do not detect leaks, we only care about UB inducing memory issues
  string(APPEND MPIEXEC_EXTRA_OPTS_FOR_TESTING " -x LSAN_OPTIONS=detect_leaks=0")
  set(FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS "LSAN_OPTIONS=detect_leaks=0")
endif()

set(FOUR_C_EXECUTABLE_NAME 4C)
set(FULL_TARGETS
    post_processor
    post_monitor
    pre_exodus
    pre_locsys
    create_rtd
    cut_test
    )

add_custom_target(full DEPENDS ${FOUR_C_EXECUTABLE_NAME} ${FULL_TARGETS})

include(cmake/setup_tests.cmake)

# Create one large library: every module adds itself to this target
set(FOUR_C_LIBRARY_NAME lib4C)
add_library(${FOUR_C_LIBRARY_NAME})
# Remove the CMake generated prefix "lib" because we already added it manually, to have a distinct target name.
set_target_properties(${FOUR_C_LIBRARY_NAME} PROPERTIES PREFIX "")
target_link_libraries(${FOUR_C_LIBRARY_NAME} PRIVATE four_c_private_compile_interface)

# Set up Python build/test venv
include(cmake/python/set_up_python.cmake)

# Read in the source directory which defines all modules
add_subdirectory(src)
add_subdirectory(apps)

# Process documentation
add_subdirectory(doc/doxygen EXCLUDE_FROM_ALL)
add_subdirectory(doc/readthedocs EXCLUDE_FROM_ALL)

###------------------------------------------------------------------ Tests

add_subdirectory(unittests)
add_subdirectory(tests/cut_test)
include(tests/list_of_tests.cmake)
