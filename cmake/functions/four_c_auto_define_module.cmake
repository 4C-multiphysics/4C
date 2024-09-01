# Automatically create a library for the sources and headers in the current directory. The target
# will be named based on the folder name. If this function is called revursively inside an already
# defined module, the sources are appended to the already defined module. The module name is returned in the variable
# AUTO_DEFINED_MODULE_NAME which is set at the call site.
function(four_c_auto_define_module)

  if("${FOUR_C_CURRENTLY_DEFINED_PARENT_MODULE}" STREQUAL "")
    # No parent module is set, so this must be the first call in the hierarchy:
    # create the necessary targets with a name based on the current folder
    get_filename_component(_target ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(VERBOSE "Defining module target: ${_target}")

    # Define an interface library for usage requirements only
    add_library(${_target}_deps INTERFACE)
    # Link against all default external libraries
    four_c_link_default_external_libraries(${_target}_deps INTERFACE)
    # Always add the special config target as a dependency
    four_c_add_dependency(${_target} config)

    # Define an object library containing the actual sources.
    # We need to add a dummy file to have at least one file in case a module does not have any compiled sources.
    add_library(${_target}_objs OBJECT ${PROJECT_SOURCE_DIR}/cmake/dummy.cpp)
    target_link_libraries(${_target}_objs PUBLIC ${_target}_deps)
    # Add all global compile settings as PRIVATE. We only want to use them to compile our own files and not force
    # them on other users of the library.
    target_link_libraries(${_target}_objs PRIVATE four_c_private_compile_interface)

    # Collect all modules in the global library
    target_link_libraries(${FOUR_C_LIBRARY_NAME} PUBLIC ${_target}_deps)
    target_link_libraries(${FOUR_C_LIBRARY_NAME} PRIVATE ${_target}_objs)
  else()
    # Targets are already defined in parent scope, so we use the same name
    set(_target "${FOUR_C_CURRENTLY_DEFINED_PARENT_MODULE}")
    message(VERBOSE "Appending to module target: ${_target}")
  endif()

  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  file(
    GLOB _sources
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS *.cpp *.cc *.c
    )
  file(
    GLOB _headers
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS *.h *.hpp
    )

  # Remove headers that only contain template instantiations
  list(
    FILTER
    _headers
    EXCLUDE
    REGEX
    "_fwd\.h(pp)?|\.inst\.[hH]"
    )

  # Check that every header includes 4C_config.hpp
  foreach(_header ${_headers})
    file(READ ${_header} _header_content)
    string(FIND "${_header_content}" "#include \"4C_config.hpp\"" _index)
    if(_index EQUAL -1)
      message(
        FATAL_ERROR
          "The file \"${_header}\" does not include \"4C_config.hpp\". Please include the file and rerun CMake."
        )
    endif()
  endforeach()

  # Add the headers as a file set to the interface library. This will automatically add the current directory as a search directory.
  # Since we append sources to a target successively, we need to add multiple file sets with a unique name. Use
  # the current directory name for this.
  string(REGEX REPLACE "[^a-zA-Z0-9]" "_" _file_set_name "${CMAKE_CURRENT_SOURCE_DIR}")
  string(TOLOWER "${_file_set_name}" _file_set_name)
  string(SUBSTRING "${_file_set_name}" 1 -1 _file_set_name)

  target_sources(
    ${_target}_deps
    INTERFACE FILE_SET
              ${_file_set_name}
              TYPE
              HEADERS
              FILES
              ${_headers}
    )

  # Add the compiled sources to the object library
  target_sources(${_target}_objs PRIVATE ${_sources})

  # Now check if there are more directories that contain CMakeLists.txt. If yes, we also add those.
  # For this action, we become the parent module of the submodules we are about to define.
  set(FOUR_C_CURRENTLY_DEFINED_PARENT_MODULE ${_target})

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

  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Simulate a "return" by setting a variable at the call site
  set(AUTO_DEFINED_MODULE_NAME
      ${_target}
      PARENT_SCOPE
      )
endfunction()
