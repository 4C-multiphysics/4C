# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

function(_four_c_internal_link_with_debug_message target link_type deps)
  message(DEBUG "Linking: ${target} <- ${deps} (${link_type})")
  target_link_libraries(${target} ${link_type} ${deps})
endfunction()

function(four_c_add_dependency target)
  # Internal target in the library
  if(TARGET ${target}_deps)
    foreach(_dep ${ARGN})
      # Skip self-linking
      if(NOT ${_dep} STREQUAL ${target})
        _four_c_internal_link_with_debug_message(${target}_deps INTERFACE ${_dep}_deps)

        if(TARGET 4C_${target})
          if(NOT TARGET 4C_${_dep})
            message(
              FATAL_ERROR
                "Trying to link 4C_${_dep} to 4C_${target} but module ${_dep} does not provide this library."
              )
          endif()
          target_link_libraries(4C_${target} PUBLIC 4C_${_dep})
        endif()
      endif()
    endforeach()
    # Some other target
  else()
    get_target_property(_target_type ${target} TYPE)
    if(${_target_type} STREQUAL INTERFACE_LIBRARY)
      foreach(_dep ${ARGN})
        _four_c_internal_link_with_debug_message(${target} INTERFACE ${_dep}_deps)
      endforeach()
    elseif(${_target_type} STREQUAL EXECUTABLE)
      # Currently the code base is not able to link without cycles. When linking an executable to a library, the
      # transitive dependencies must not contain any cycles. For static libraries this can be ignored when the
      # libraries are repeated on the link line. However, for shared libraries there is no such workaround.
      # For this reason, we issue an error here (for now).
      message(
        FATAL_ERROR "Trying to link parts of the library to ${target} which is an executable. "
                    "Please link against the whole library."
        )
    else()
      message(FATAL_ERROR "Cannot add dependency to ${target} of type ${_target_type}.")
    endif()
  endif()
endfunction()

function(four_c_add_external_dependency target)
  # Internal target in the library: adjust name
  if(TARGET ${target}_deps)
    set(target ${target}_deps)
  endif()
  get_target_property(_target_type ${target} TYPE)
  if(${_target_type} STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(${target} INTERFACE ${ARGN})
  else()
    target_link_libraries(${target} PUBLIC ${ARGN})
  endif()
endfunction()
