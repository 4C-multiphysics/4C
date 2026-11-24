# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

### Function to execute a process and check for errors
function(_execute_process)
  set(options "")
  set(oneValueArgs HINT)
  set(multiValueArgs PROCESS_COMMAND)
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

  if(NOT DEFINED _parsed_PROCESS_COMMAND)
    message(FATAL_ERROR "Process command is a necessary input argument!")
  endif()

  execute_process(
    COMMAND ${_parsed_PROCESS_COMMAND}
    RESULT_VARIABLE _command_result
    OUTPUT_VARIABLE _command_output
    ERROR_VARIABLE _command_error
    )

  if(NOT _command_result EQUAL "0")
    message(
      STATUS
        "Failed to execute command '${_parsed_PROCESS_COMMAND}' during setup of virtual Python environment for building and testing!"
      )
    message(STATUS "Output: ${_command_output}")
    message(STATUS "Result: ${_command_result}")
    message(FATAL_ERROR "Error: ${_command_error}\n${_parsed_HINT}")
  endif()
endfunction()

### Set up a virtual Python environment for building and testing 4C
message(STATUS "Setting up virtual Python environment for building and testing 4C")
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(Python_FOUND)
  message(STATUS "Using python executable: ${Python_EXECUTABLE} (V${Python_VERSION})")
endif()

if(Python_VERSION VERSION_LESS "3.12")
  message(
    FATAL_ERROR
      "Python version must be at least 3.12, but found ${Python_VERSION}. "
      "Please install a newer version of Python and set FOUR_C_PYTHON_ROOT to the correct path."
    )
endif()

set(FOUR_C_PYTHON_VENV_BUILD "${PROJECT_BINARY_DIR}/python_venv_build_test")
set(FOUR_C_PYTHON_VENV_BUILD
    "${PROJECT_BINARY_DIR}/python_venv_build_test"
    PARENT_SCOPE
    )

four_c_process_global_option(
  FOUR_C_PYTHON_SKIP_INSTALL
  DESCRIPTION
  "Reuse the existing virtual Python environment in the build directory without (re)installing packages, e.g., to reconfigure without Internet access"
  DEFAULT
  OFF
  )

if(FOUR_C_PYTHON_SKIP_INSTALL)
  if(NOT EXISTS "${FOUR_C_PYTHON_VENV_BUILD}/bin/python3")
    message(
      FATAL_ERROR
        "FOUR_C_PYTHON_SKIP_INSTALL is enabled, but no virtual Python environment exists in ${FOUR_C_PYTHON_VENV_BUILD}. "
        "Configure once with FOUR_C_PYTHON_SKIP_INSTALL=OFF to create it."
      )
  endif()
  message(
    STATUS
      "FOUR_C_PYTHON_SKIP_INSTALL is enabled: reusing existing virtual Python environment in ${FOUR_C_PYTHON_VENV_BUILD} without installing packages"
    )
else()
  _execute_process(
    PROCESS_COMMAND
    ${Python_EXECUTABLE}
    -m
    venv
    ${FOUR_C_PYTHON_VENV_BUILD}
    )
  _execute_process(
    PROCESS_COMMAND
    ${FOUR_C_PYTHON_VENV_BUILD}/bin/pip
    install
    -e
    ${PROJECT_SOURCE_DIR}/utilities/four_c_python[build]
    HINT
    "If you have no Internet access and the virtual Python environment was set up before, reconfigure with -DFOUR_C_PYTHON_SKIP_INSTALL=ON to reuse it."
    )
endif()

# determine site-packages of build environment
execute_process(
  COMMAND
    "${FOUR_C_PYTHON_VENV_BUILD}/bin/python3" -c
    "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
# Export site-packages to the parent scope
set(FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES
    "${FOUR_C_PYTHON_VENV_BUILD_SITE_PACKAGES}"
    PARENT_SCOPE
    )

if(NOT FOUR_C_PYTHON_SKIP_INSTALL)
  message(
    STATUS
      "Successfully created Python virtual environment in ${FOUR_C_PYTHON_VENV_BUILD} and installed necessary requirements"
    )
endif()
