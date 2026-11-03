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

set(_four_c_python_build_install_inputs
    "${PROJECT_SOURCE_DIR}/utilities/four_c_python/pyproject.toml"
    "${PROJECT_SOURCE_DIR}/utilities/four_c_python/requirements.txt"
    "${PROJECT_SOURCE_DIR}/utilities/four_c_python/build-requirements.txt"
    )
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${_four_c_python_build_install_inputs}
  )

set(_four_c_python_build_install_fingerprint
    "${PROJECT_SOURCE_DIR};${Python_EXECUTABLE};${Python_VERSION}"
    )
foreach(_input_file IN LISTS _four_c_python_build_install_inputs)
  file(SHA256 "${_input_file}" _input_hash)
  string(APPEND _four_c_python_build_install_fingerprint ";${_input_hash}")
endforeach()
string(
  SHA256 _four_c_python_build_install_fingerprint "${_four_c_python_build_install_fingerprint}"
  )

# The source checkout and package inputs are fingerprinted; the venv is mutable.
set(_four_c_python_build_install_stamp "${PROJECT_BINARY_DIR}/four_c_python_build_install_stamp")
set(_four_c_python_build_install_is_current FALSE)
if(EXISTS "${_four_c_python_build_install_stamp}"
   AND EXISTS "${FOUR_C_PYTHON_VENV_BUILD}/bin/python3"
   )
  file(READ "${_four_c_python_build_install_stamp}" _installed_fingerprint)
  if(_installed_fingerprint STREQUAL _four_c_python_build_install_fingerprint)
    set(_four_c_python_build_install_is_current TRUE)
  endif()
endif()

if(_four_c_python_build_install_is_current)
  set(_four_c_python_update_report "${PROJECT_BINARY_DIR}/four_c_python_update_report.json")
  file(REMOVE "${_four_c_python_update_report}")
  execute_process(
    COMMAND
      "${FOUR_C_PYTHON_VENV_BUILD}/bin/python3" -m pip install --dry-run --upgrade
      --upgrade-strategy eager --report "${_four_c_python_update_report}" --retries 2 --timeout 3
      --disable-pip-version-check -e "${PROJECT_SOURCE_DIR}/utilities/four_c_python[build]"
    RESULT_VARIABLE _four_c_python_update_check_result
    OUTPUT_QUIET ERROR_QUIET
    )
  if(_four_c_python_update_check_result EQUAL 0)
    file(READ "${_four_c_python_update_report}" _four_c_python_update_report_contents)
    string(
      JSON _four_c_python_update_count ERROR_VARIABLE _four_c_python_report_error
      LENGTH "${_four_c_python_update_report_contents}" install
      )
    file(REMOVE "${_four_c_python_update_report}")
    if(NOT _four_c_python_report_error STREQUAL "NOTFOUND")
      message(FATAL_ERROR "Could not parse pip update report: ${_four_c_python_report_error}")
    endif()

    set(_four_c_python_updates_available FALSE)
    if(_four_c_python_update_count GREATER 0)
      math(EXPR _four_c_python_last_update_index "${_four_c_python_update_count} - 1")
      foreach(_update_index RANGE 0 ${_four_c_python_last_update_index})
        string(
          JSON
          _update_name
          GET
          "${_four_c_python_update_report_contents}"
          install
          ${_update_index}
          metadata
          name
          )
        string(TOLOWER "${_update_name}" _update_name)
        if(NOT _update_name STREQUAL "four_c_python")
          set(_four_c_python_updates_available TRUE)
        endif()
      endforeach()
    endif()
    if(_four_c_python_updates_available)
      message(STATUS "Python dependency updates are available; updating build environment")
      file(REMOVE "${_four_c_python_build_install_stamp}")
      _execute_process(
        PROCESS_COMMAND
        "${FOUR_C_PYTHON_VENV_BUILD}/bin/python3"
        -m
        pip
        install
        --upgrade
        --upgrade-strategy
        eager
        -e
        "${PROJECT_SOURCE_DIR}/utilities/four_c_python[build]"
        )
      file(
        WRITE "${_four_c_python_build_install_stamp}" "${_four_c_python_build_install_fingerprint}"
        )
    else()
      message(STATUS "Python build environment is up to date")
    endif()
  else()
    file(REMOVE "${_four_c_python_update_report}")
    message(
      WARNING "Could not check for Python dependency updates; using existing build environment"
      )
  endif()
else()
  file(REMOVE "${_four_c_python_build_install_stamp}")
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
    --upgrade
    --upgrade-strategy
    eager
    -e
    ${PROJECT_SOURCE_DIR}/utilities/four_c_python[build]
    )
  file(WRITE "${_four_c_python_build_install_stamp}" "${_four_c_python_build_install_fingerprint}")
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

message(STATUS "Successfully configured Python virtual environment in ${FOUR_C_PYTHON_VENV_BUILD}")
