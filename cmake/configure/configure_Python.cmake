# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

### Function to execute a process and check for errors
function(_execute_process)
  set(options "")
  set(oneValueArgs "")
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
    message(FATAL_ERROR "Error: ${_command_error}")
  endif()
endfunction()

### Function to check for dependencies of the venv identifier
function(_check_deps OUT_STATUS PYTHON_EXEC REQ_SPEC)
  # Run inline Python script to inspect installed dependencies in the venv
  execute_process(
    COMMAND
      "${PYTHON_EXEC}" -c [=[
import sys, re
from importlib.metadata import distribution, PackageNotFoundError
from packaging.requirements import Requirement

spec = sys.argv[1]
match = re.match(r"^([^\[]+)(?:\[([^\]]+)\])?$", spec)
if not match:
    sys.exit(1)

pkg_name, extras_str = match.groups()
requested_extras = [e.strip() for e in extras_str.split(",")] if extras_str else []

try:
    dist = distribution(pkg_name)
except PackageNotFoundError:
    # Package itself is not installed yet
    sys.exit(1)

# If no extras were requested, the base package presence is enough
if not requested_extras:
    sys.exit(0)

# Check dependencies tied to the requested extra(s)
requires_list = dist.requires or []
for req_str in requires_list:
    req = Requirement(req_str)

    # Check if this dependency belongs to any of the requested extras
    for extra in requested_extras:
        if req.marker and req.marker.evaluate({"extra": extra}):
            try:
                dep_dist = distribution(req.name)
                # Verify version constraints if specified
                if req.specifier and not req.specifier.contains(dep_dist.version, prereleases=True):
                    sys.exit(1)
            except PackageNotFoundError:
                sys.exit(1)

sys.exit(0) # Everything is satisfied
]=] "${REQ_SPEC}"
    RESULT_VARIABLE _STATUS
    ERROR_QUIET
    )

  # Propagate the result back to the caller's variable
  set(${OUT_STATUS}
      ${_STATUS}
      PARENT_SCOPE
      )
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

_execute_process(
  PROCESS_COMMAND
  ${Python_EXECUTABLE}
  -m
  venv
  ${FOUR_C_PYTHON_VENV_BUILD}
  )

find_program(
  FOUR_C_VENV_PYTHON
  NAMES python python3
  PATHS "${FOUR_C_PYTHON_VENV_BUILD}/bin"
  NO_DEFAULT_PATH
  )

# Construct the pip flags based on whether dependencies are already satisfied
_check_deps(DEPS_STATUS "${FOUR_C_VENV_PYTHON}" "four_c_python[build]")

if(DEPS_STATUS EQUAL 0)
  message(STATUS "four_c_python[build] dependencies already satisfied in venv.")
else()
  message(
    STATUS
      "four_c_python[build] missing dependencies detected. Attempting pip install (require Internet connection)."
    )

  # Execute pip install
  _execute_process(
    PROCESS_COMMAND
    "${FOUR_C_VENV_PYTHON}"
    -m
    pip
    install
    -e
    "${PROJECT_SOURCE_DIR}/utilities/four_c_python[build]"
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

message(
  STATUS
    "Successfully created Python virtual environment in ${FOUR_C_PYTHON_VENV_BUILD} and installed necessary requirements"
  )
