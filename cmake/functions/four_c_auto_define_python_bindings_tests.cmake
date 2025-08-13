# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(four_c_auto_define_python_bindings_tests)
  # only add tests if
  if(NOT FOUR_C_ENABLE_PYTHON_BINDINGS)
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
    message(FATAL_ERROR "MODULE not set")
  endif()

  set(_test_directory ${PROJECT_BINARY_DIR}/python_binding_test_${_parsed_MODULE})

  set(_python_binding_test_command_list
      "rm -rf ${_test_directory}" "mkdir -p ${_test_directory}" "cd ${_test_directory}"
      "python -m venv venv" ". venv/bin/activate" "pip install --upgrade pip"
      "[ -f ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt ] && pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
      "pip install pytest"
      "pip install -e ${PROJECT_BINARY_DIR}/${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}" # Install bindings in developer mode!
      "pytest ${CMAKE_CURRENT_SOURCE_DIR}" "rm -rf ${_test_directory}"
      )

  # Join them into a single shell command separated by &&
  string(JOIN " && " _python_binding_test_command ${_python_binding_test_command_list})

  add_test(
    NAME ${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}.${_parsed_MODULE}
    COMMAND ${CMAKE_COMMAND} -E env bash -c "${_python_binding_test_command}"
    )

endfunction()
