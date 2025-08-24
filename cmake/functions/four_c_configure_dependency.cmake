# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# This function checks for the existence of a file dependencies/supported_version/${_package_name}.txt and reports
# whether the given hash is supported.
function(four_c_check_dependency_version _package_name _package_name_sanitized _sha)
  # First check if we have a user-overridden version.
  if(DEFINED "FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION")
    message(
      STATUS
        "You specified to treat your version of ${_package_name} (hash ${_sha}) as internally numbered version "
        "${FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION}."
      )
    string(
      REGEX MATCH
            "^([0-9]+)\\.([0-9]+)$"
            _dummy
            ${FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION}
      )
    if(NOT _dummy)
      message(
        FATAL_ERROR
          "The version number ${FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION} you supplied for ${_package_name} is not of the form 'major.minor'."
        )
    endif()
    set(_major ${CMAKE_MATCH_1})
    set(_minor ${CMAKE_MATCH_2})

    set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR
        ${_major}
        PARENT_SCOPE
        )
    set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR
        ${_minor}
        PARENT_SCOPE
        )
    return()
  endif()
  if(EXISTS "${PROJECT_SOURCE_DIR}/dependencies/supported_version/${_package_name}.txt")
    # File has multiple lines each of the form: <hash> <internal_version> <optional_external_version>
    file(
      STRINGS
      "${PROJECT_SOURCE_DIR}/dependencies/supported_version/${_package_name}.txt"
      _supported_versions
      )
    # If the file is empty, we cannot check for compatibility.
    if(NOT _supported_versions)
      message(
        WARNING "The file 'dependencies/supported_version/${_package_name}.txt' is empty. "
                "Cannot check for compatibility."
        )
      return()
    endif()
    set(_index 0)
    foreach(_supported_version ${_supported_versions})
      # Split on whitespace and name the parts.
      separate_arguments(_supported_version_split UNIX_COMMAND ${_supported_version})
      list(
        POP_FRONT
        _supported_version_split
        _supported_sha
        _supported_internal_version
        _supported_external_version
        )
      # Sanity check: is the supported version a major.minor version?
      if(NOT _supported_internal_version MATCHES "^[0-9]+\\.[0-9]+$")
        message(
          FATAL_ERROR
            "File 'dependencies/supported_version/${_package_name}.txt' contains an unsupported version format: ${_supported_internal_version}. "
            "Needs to be of the form 'major.minor'."
          )
      endif()
      if(${_index} EQUAL 0)
        set(_latest_version ${_supported_internal_version})
      endif()

      # Check if the hash matches the beginning of the first element as we might have a short _sha.
      string(FIND ${_supported_sha} ${_sha} _sha_position)
      if(${_sha_position} EQUAL 0)
        message(
          STATUS
            "Identified 4C-internal version ${_supported_internal_version} of ${_package_name} (hash ${_sha}, package's own version name: ${_supported_external_version})."
          )
        if(${_index} EQUAL 0)
          message(STATUS "This is the latest supported version of ${_package_name}.")
        else()
          message(
            STATUS
              "NOTE: This is not the latest supported version of ${_package_name}. Check the file 'dependencies/supported_version/${_package_name}.txt' for more information."
            )
        endif()
        set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION
            ${_supported_internal_version}
            PARENT_SCOPE
            )

        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)$" _dummy ${_supported_internal_version})
        set(_major ${CMAKE_MATCH_1})
        set(_minor ${CMAKE_MATCH_2})

        set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION
            ${_supported_internal_version}
            PARENT_SCOPE
            )
        set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR
            ${_major}
            PARENT_SCOPE
            )
        set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR
            ${_minor}
            PARENT_SCOPE
            )
        return()
      endif()
      math(EXPR _index "${_index} + 1")
    endforeach()
    message(STATUS "")
    message(STATUS "Could not identify your version of ${_package_name} with hash ${_sha}.")
    message(STATUS "Supported versions are:")
    foreach(_supported_version ${_supported_versions})
      separate_arguments(_supported_version_split UNIX_COMMAND ${_supported_version})
      list(
        POP_FRONT
        _supported_version_split
        _supported_sha
        _supported_internal_version
        _supported_external_version
        )
      message(
        STATUS
          "  - ${_supported_internal_version} (hash ${_supported_sha}, package's own version name: ${_supported_external_version})"
        )
    endforeach()
    message(
      STATUS "Assume your version is newer than the latest supported version ${_latest_version}."
      )
    # Split latest version into major and minor. Increment minor version by 1.
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)$" _dummy ${_latest_version})
    set(_major ${CMAKE_MATCH_1})
    set(_minor ${CMAKE_MATCH_2})
    math(EXPR _minor "${_minor} + 1")

    set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION
        "${_major}.${_minor}"
        PARENT_SCOPE
        )
    set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR
        ${_major}
        PARENT_SCOPE
        )
    set(FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR
        ${_minor}
        PARENT_SCOPE
        )

    # Make an assumption that the package is newer than the latest supported version.
    message(
      WARNING
        "Your version of package ${_package_name} with hash ${_sha} could not be identified as a supported version. "
        "Please check the supported versions in the file 'dependencies/supported_version/${_package_name}.txt'. "
        "I made the assumption that the package is newer than the latest supported version and gave it the "
        "internal version number ${_major}.${_minor}. "
        "If this assumption is wrong, please provide the correct version information in the cache variable "
        "'FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION'."
      )
  else()
    message(
      STATUS "No supported versions known for ${_package_name}. Skipping compatibility check."
      )
  endif()
endfunction()

# A function to search for and configure an external dependency.
#
# Usage: four_c_configure_dependency(<package_name> [REQUIRED] [DEFAULT <ON|OFF>])
#
# Note that setting the option 'REQUIRED' will imply `DEFAULT ON`.
#
# This function takes over most of the boilerplate to generate unified names for the dependencies. The only thing
# it requires, is a file called 'configure_${_package_name}.cmake' in the 'cmake/configure' directory. This file
# is responsible to do whatever is necessary to find and configure the package. The configure script can use the
# following variables on input:
#
# - ${_package_name}_ROOT: The root directory of the package.
#
# The configure script should set the following variables:
#
# - FOUR_C_${_package_name}_GIT_HASH: The git hash of the package. (Optional, but may be used to check supported versions.)
function(four_c_configure_dependency _package_name)
  set(options REQUIRED)
  set(oneValueArgs DEFAULT)
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

  # Sanitize DEFAULT and REQUIRED options to be consistent
  if(_parsed_REQUIRED)
    if(NOT DEFINED _parsed_DEFAULT)
      set(_parsed_DEFAULT "ON")
    elseif(NOT _parsed_DEFAULT)
      message(
        FATAL_ERROR
          "You cannot set the option 'DEFAULT OFF' for 'REQUIRED' dependency '${_package_name}'. "
        )
    endif()
  else()
    if(NOT DEFINED _parsed_DEFAULT)
      message(
        FATAL_ERROR "You must specify a default value for the dependency '${_package_name}'. "
                    "Use 'DEFAULT ON' or 'DEFAULT OFF'."
        )
    endif()
  endif()

  # Sanitize the package name: all upper case, no hyphens and dots.
  string(TOUPPER ${_package_name} _package_name_sanitized)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _package_name_sanitized ${_package_name_sanitized})

  # Add a cache entry to turn the option ON or OFF.
  four_c_process_global_option(
    FOUR_C_WITH_${_package_name_sanitized}
    DESCRIPTION
    "Build 4C with ${_package_name}"
    DEFAULT
    ${_parsed_DEFAULT}
    )
  # Add a cache entry to set the root directory of the package.
  four_c_process_cache_variable(
    FOUR_C_${_package_name_sanitized}_ROOT
    TYPE
    PATH
    DESCRIPTION
    "Root directory of ${_package_name}"
    DEFAULT
    ""
    )

  if(FOUR_C_WITH_${_package_name_sanitized})
    if(${_package_name}_ROOT)
      message(
        WARNING
          "A variable '${_package_name}_ROOT' is set. Prefer setting it via 'FOUR_C_${_package_name_sanitized}_ROOT'."
        )
    endif()

    if(FOUR_C_${_package_name_sanitized}_ROOT)
      # Translate the ROOT variable into the case style that fits to the package name.
      # This variable is automatically understood by CMake's find_XXX functions.
      set(${_package_name}_ROOT ${FOUR_C_${_package_name_sanitized}_ROOT})
    else()
      message(
        STATUS
          "No variable 'FOUR_C_${_package_name_sanitized}_ROOT' set. Trying to find ${_package_name} in default locations."
        )
    endif()

    # Hand over the actual configuration to a separate script.
    include(cmake/configure/configure_${_package_name}.cmake)

    string(APPEND _configuration_string "#define FOUR_C_WITH_${_package_name_sanitized}\n")

    # Now check if the configure script has set the hash variable.
    if(DEFINED FOUR_C_${_package_name}_GIT_HASH)
      four_c_check_dependency_version(
        ${_package_name} ${_package_name_sanitized} ${FOUR_C_${_package_name}_GIT_HASH}
        )
      if(DEFINED FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR)
        string(
          APPEND
          _configuration_string
          "#define FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR ${FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR}\n"
          "#define FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR ${FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR}\n"
          # Define a macro to check if the version is greater or equal to a given version.
          "#define FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_GE(major, minor) \\\n"
          "  (FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MAJOR * 1000 + \\\n"
          "  FOUR_C_${_package_name_sanitized}_INTERNAL_VERSION_MINOR\\\n"
          "  >= ((major) * 1000 + (minor)))\n"
          "#define FOUR_C_${_package_name_sanitized}_HASH \"${FOUR_C_${_package_name}_GIT_HASH}\"\n"
          )
      endif()
    else()
      message(
        STATUS "No version information for ${_package_name} detected. Skipping compatibility check."
        )
    endif()
  else()
    string(APPEND _configuration_string "/* #undef FOUR_C_WITH_${_package_name_sanitized} */\n")
  endif()

  string(APPEND _configuration_string "\n")
  set_property(
    GLOBAL APPEND_STRING PROPERTY FOUR_C_DEFINES_FOR_EXTERNAL_DEPENDENCIES ${_configuration_string}
    )

  message(STATUS "Processed dependency ${_package_name}.\n")
endfunction()
