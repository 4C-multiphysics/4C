# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Automatically creates a python binding submodule for all sources and headers in the current directory. The target
# will be named based on the folder name. If this function is called recursively inside an already
# defined submodule, the sources are appended to the already defined submodule. The submodule name is returned
# in the variable AUTO_DEFINED_SUBMODULE_NAME which is set at the call site.
function(four_c_auto_define_python_binding_submodule)
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
    set(_bindings_for_module ${_parsed_MODULE})
  else()
    if("${FOUR_C_CURRENTLY_DEFINED_PARENT_SUBMODULE}" STREQUAL "")
      message(
        FATAL_ERROR
          "No parent module is set. Either give the module these bindings belongs to or call this functions inside a module."
        )
    endif()

    set(_bindings_for_module "${FOUR_C_CURRENTLY_DEFINED_PARENT_SUBMODULE}")
  endif()

  if(NOT TARGET "${_bindings_for_module}_objs")
    message(
      FATAL_ERROR
        "Tried to add bindings for a module named '${_bindings_for_module}' which is not a known module name."
      )
  endif()

  # create the name of the current binding
  set(_target pybind11_${_bindings_for_module})

  # create an object target for the current submodule (and fill it with a dummy cpp-file)
  add_library(${_target}_objs OBJECT ${PROJECT_SOURCE_DIR}/cmake/dummy.cpp)

  # Add all source files of this folder to this target
  file(GLOB_RECURSE _sources CONFIGURE_DEPENDS *.cpp)
  target_sources(${_target}_objs PRIVATE ${_sources})

  # Link the python binding configuration to this target
  target_link_libraries(${_target}_objs PUBLIC py4C_config)

  # Python bindings require position independent code so that Python can dynamically link the library
  target_link_libraries(${_target}_objs PRIVATE four_c_private_compile_interface)

  # Link all dependencies of the respective 4C module to this python binding submodule
  target_link_libraries(${_target}_objs PRIVATE ${_bindings_for_module}_module)

  # Add the current submodule object file to the main python bindings target
  target_link_libraries(${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME} PRIVATE ${_target}_objs)

  # Now check if there are more directories that contain CMakeLists.txt. If yes, we also add those.
  # For this action, we become the parent submodule of the sub-submodules we are about to define.
  set(FOUR_C_CURRENTLY_DEFINED_PARENT_SUBMODULE ${_target})
  # Recursively add all subdirectories that contain CMakeLists.txt files.
  # N.B. We need to directly glob for CMakeLists.txt files here to ensure
  # the glob reruns when a new CMakeLists.txt is added.
  file(
    GLOB children
    RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/*/CMakeLists.txt
    )
  foreach(child ${children})
    get_filename_component(_subdir ${child} DIRECTORY)
    add_subdirectory(${_subdir})
  endforeach()

  # Simulate a "return" by setting a variable at the call site
  set(AUTO_DEFINED_SUBMODULE_NAME
      ${_target}
      PARENT_SCOPE
      )
endfunction()
