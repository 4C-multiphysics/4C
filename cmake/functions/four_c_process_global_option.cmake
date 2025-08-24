# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Declare an option with given name, description and default value (ON or OFF).
# Options need to start with "FOUR_C_".
#
# Usage:
#   four_c_process_global_option(FOUR_C_OPTION_NAME
#       DESCRIPTION "Description of the option"
#       DEFAULT "ON|OFF"
#       [DEPRECATED_NAMES "NAME1" "NAME2"])
#
function(four_c_process_global_option option_name)
  if(NOT option_name MATCHES "FOUR_C_.*")
    message(FATAL_ERROR "Disallowed option '${option_name}'. Option needs to start with 'FOUR_C_'.")
  endif()

  set(options "")
  set(oneValueArgs DESCRIPTION DEFAULT)
  set(multiValueArgs DEPRECATED_NAMES)
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

  if(NOT _parsed_DESCRIPTION)
    message(FATAL_ERROR "Description for option '${option_name}' is not set.")
  endif()

  if(NOT _parsed_DEFAULT MATCHES "ON|OFF")
    message(FATAL_ERROR "Default value for option '${option_name}' must be ON or OFF.")
  endif()

  # Check if any of the deprecated names are set. Warn if so and set the option to the new name.
  if(DEFINED _parsed_DEPRECATED_NAMES)
    foreach(deprecated_name ${_parsed_DEPRECATED_NAMES})
      if(DEFINED ${deprecated_name})
        message(
          WARNING
            "Option '${deprecated_name}' is deprecated. Please use '${option_name}' instead.\n"
            "I will now set '${option_name}' to the value of '${deprecated_name}', i.e., ${${deprecated_name}}."
          )
        set(${option_name}
            ${${deprecated_name}}
            CACHE BOOL "${_parsed_DESCRIPTION} (default: ${_parsed_DEFAULT})" FORCE
            )
        unset(${deprecated_name} CACHE)
      endif()
    endforeach()
  endif()

  option(
    "${option_name}" "${_parsed_DESCRIPTION} (default: ${_parsed_DEFAULT})" "${_parsed_DEFAULT}"
    )
  if(${option_name})
    message(STATUS "Option ${option_name} = ON")
  else()
    message(STATUS "Option ${option_name} = OFF")
  endif()
endfunction()

# Initialize a cache variable. This function is almost equivalent to the builtin CMake
# set() command, except that it also prints info on whether the variable is set and allows to
# ensure a consistent style. Note that you need to use four_c_process_global_option() for the
# common case of a BOOL variable.
#
# Usage:
#   four_c_initialize_cache_variable(variable_name
#     TYPE <STRING|FILEPATH|PATH>
#     DESCRIPTION "Description of the variable"
#     DEFAULT "Default value of the variable")
#
function(four_c_process_cache_variable variable_name)
  if(NOT variable_name MATCHES "FOUR_C_.*")
    message(
      FATAL_ERROR
        "Disallowed variable name '${variable_name}'. Variable name needs to start with 'FOUR_C_'."
      )
  endif()

  set(options "")
  set(oneValueArgs TYPE DESCRIPTION DEFAULT)
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

  # Ensure that only one of the options STRING or (FILE)PATH for the cache variable type is set.
  if(NOT _parsed_TYPE MATCHES "STRING|FILEPATH|PATH")
    message(
      FATAL_ERROR
        "Invalid type '${_parsed_TYPE}' for cache variable '${variable_name}'. Allowed types are STRING, FILEPATH and PATH."
      )
  endif()

  set(${variable_name}
      "${_parsed_DEFAULT}"
      CACHE ${_parsed_TYPE} "${_parsed_DESCRIPTION} (default: ${_parsed_DEFAULT})"
      )
  if(${variable_name} STREQUAL "")
    message(STATUS "Cache variable ${variable_name} not set")
  else()
    message(STATUS "Cache variable ${variable_name} = ${${variable_name}}")
  endif()
endfunction()
