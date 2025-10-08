# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#
# Set up the main documentation based on Sphinx

four_c_process_global_option(
  FOUR_C_ENABLE_DOCUMENTATION
  DESCRIPTION
  "Build the documentation"
  DEFAULT
  OFF
  )
if(FOUR_C_ENABLE_DOCUMENTATION)

  if(NOT FOUR_C_WITH_PYTHON)
    message(FATAL_ERROR "FOUR_C_ENABLE_DOCUMENTATION requires FOUR_C_WITH_PYTHON to be enabled.")
  endif()

  # add doc-specific cmake modules
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # set input and output files and directories
  set(_sphinx_SOURCE_DIR ${PROJECT_SOURCE_DIR}/doc/documentation)
  set(_sphinx_TPLDEPEND_DIR ${PROJECT_SOURCE_DIR}/dependencies/current)
  set(_sphinx_CONFIG_IN ${_sphinx_SOURCE_DIR}/conf.py.in)
  set(_sphinx_OUT_DIR ${PROJECT_BINARY_DIR}/doc/documentation)
  set(_sphinx_CONFIG_OUT ${_sphinx_OUT_DIR}/conf.py)
  set(_sphinx_HTML_DIR ${_sphinx_OUT_DIR}/html)

  # Define some extra files that are to be copied to the build directory
  # This is necessary, since some of the input files are needed for the documentation
  # (and we don't want to copy all of them blindly)
  set(_sphinx_TEST_INPUT_FILES
      ${PROJECT_SOURCE_DIR}/tests/input_files/solid_runtime_hex8.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/xfluid_beltrami_stat_16x16x16_hk_vol_equivalent.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/sohex8_disp_altgeogeneration.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/f3_forced_hit_geogen_hdg.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/fbi_fluid_obstacle.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/tsi_lincompression_iterstaggtemp.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/porofluid_pressure_based_2D_quad4.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/contact2D_centrdiff.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/contact2D_multibody.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/contact3D_unbiased_self_hex8_nitsche.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/meshtying2D_inf_plate.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/contact3D_slidingblock_penalty_new_struct.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/tsi_locsys_monolithic.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/prestress_new.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/one_d_3_artery_network.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/sohex8_dbc_monitor.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/constr3D_NormComp_STI.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/constr2D_MPC_angle.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/elch_gaussian_hill_pbc_genalpha.4C.yaml
      ${PROJECT_SOURCE_DIR}/tests/input_files/rve3d_periodic_bcs.4C.yaml
      )

  set(_sphinx_FILES_FROM_TESTS_DIR "${_sphinx_OUT_DIR}/testfiles")
  file(GLOB _sphinx_YAML_FILES "${PROJECT_SOURCE_DIR}/tests/tutorials/*.4C.yaml")
  file(GLOB _sphinx_JOU_FILES "${PROJECT_SOURCE_DIR}/tests/tutorials/*.jou")

  #install the python virtual environment with sphinx
  find_package(Sphinx REQUIRED)

  set(_sphinx_THEME sphinx_rtd_theme)

  configure_file(${_sphinx_CONFIG_IN} ${_sphinx_CONFIG_OUT} @ONLY)
  message("Sphinx configure file conf.py is written based on ${_sphinx_CONFIG_IN}")

  add_custom_target(
    documentation ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/doc/tmp"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_sphinx_OUT_DIR}/reference_docs"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_sphinx_HTML_DIR}"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_sphinx_OUT_DIR}/testfiles"
    # create_rtd is used to generate some element information
    # that is later used to create element plots for the documentation.
    # It also creates the directory ${_sphinx_OUT_DIR}/reference_docs again
    COMMAND create_rtd
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/testfiles"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/doc/tmp"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/tutorials"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/analysis_guide"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/reference_images"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/reference_guide"
    # Copy files specified above from _sphinx_TEST_INPUT_FILES
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${_sphinx_TEST_INPUT_FILES}
      "${_sphinx_OUT_DIR}/testfiles"
      # Copy tutorial files
    COMMAND
      ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/tests/tutorials"
      "${_sphinx_OUT_DIR}/testfiles"
      # Run 4C to generate a help file
    COMMAND ${FOUR_C_EXECUTABLE_NAME} -h > ${_sphinx_OUT_DIR}/reference_docs/4C-help.txt
    # Generate a cmake preset list which is used in the documentation
    COMMAND
      ${CMAKE_COMMAND} ${PROJECT_SOURCE_DIR} --list-presets >
      ${_sphinx_OUT_DIR}/reference_docs/4C-cmake-presets.txt
      # A list of cmake variables will also be given in the documentation.
    COMMAND
      bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_cmake_variables.sh
      ${PROJECT_BINARY_DIR}/CMakeCache.txt >
      ${_sphinx_OUT_DIR}/reference_docs/4C-cmake-variables.csv
    COMMAND
      ${SPHINX_EXECUTABLE} -W -n -q -b html # warnings are errors and be nit-picky
      -i "${CMAKE_CURRENT_SOURCE_DIR}/src" -i "${_sphinx_OUT_DIR}/reference_docs" -i
      "${_sphinx_FILES_FROM_TESTS_DIR}" -i "${_sphinx_TPLDEPEND_DIR}" -o "${_sphinx_HTML_DIR}" -c
      "${_sphinx_OUT_DIR}" -s "${PROJECT_BINARY_DIR}/doc/tmp"
    DEPENDS create_rtd ${FOUR_C_EXECUTABLE_NAME}
    COMMENT "Build documentation"
    )

  list(REMOVE_ITEM CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()
