# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#
# Build routine/test for doxygen documentation
four_c_process_global_option(
  FOUR_C_ENABLE_DOXYGEN
  DESCRIPTION
  "Build doxygen documentation"
  DEFAULT
  OFF
  DEPRECATED_NAMES
  "FOUR_C_BUILD_DOXYGEN"
  )
four_c_process_global_option(
  FOUR_C_DOXYGEN_USE_LOCAL_MATHJAX
  DESCRIPTION
  "Use local MathJax installation"
  DEFAULT
  OFF
  )
four_c_process_cache_variable(
  FOUR_C_DOXYGEN_LOCAL_MATHJAX_BASEPATH
  TYPE
  PATH
  DESCRIPTION
  "Path to local MathJax installation"
  DEFAULT
  ""
  )
if(FOUR_C_ENABLE_DOXYGEN)

  # add doc-specific cmake modules
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  find_package(Doxygen REQUIRED dot)

  # set input and output files
  set(DOXYGEN_CONFIG_IN ${PROJECT_SOURCE_DIR}/doc/doxygen/Doxyfile.in)
  set(DOXYGEN_OUT_DIRECTORY ${PROJECT_BINARY_DIR}/doc/doxygen)
  set(DOXYGEN_CONFIG_OUT ${DOXYGEN_OUT_DIRECTORY}/Doxyfile)

  # find mathjax
  find_package(MathJax REQUIRED)

  # get git revision information for inclusion in Doxygen
  four_c_get_git_revision_information()

  # configure doxygen configuration
  configure_file(${DOXYGEN_CONFIG_IN} ${DOXYGEN_CONFIG_OUT} @ONLY)

  add_custom_target(
    doxygen ALL
    COMMAND Doxygen::doxygen ${DOXYGEN_CONFIG_OUT}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    VERBATIM
    COMMENT "Build doxygen"
    )

  add_dependencies(doxygen setup_mathjax)

  list(REMOVE_ITEM CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()
