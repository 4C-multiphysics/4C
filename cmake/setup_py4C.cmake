# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

if(FOUR_C_ENABLE_PYTHON_BINDINGS)
  # Python bindings require position independent code
  if(NOT FOUR_C_BUILD_SHARED_LIBS)
    message(
      FATAL_ERROR
        "4C Python bindings require to build 4C with shared libraries (FOUR_C_BUILD_SHARED_LIBS)."
      )
  endif()

  # Python bindings require pybind11
  if(NOT FOUR_C_WITH_PYBIND11)
    message(
      FATAL_ERROR "4C Python bindings require to build 4C with pybind11 (FOUR_C_WITH_PYBIND11)."
      )
  endif()

  # define the name of the python module
  set(FOUR_C_PYTHON_BINDINGS_PROJECT_NAME py4C)

  # Add setup.py to make python package installable
  configure_file(
    ${PROJECT_SOURCE_DIR}/utilities/py4C/src/config/setup.py.in
    ${PROJECT_BINARY_DIR}/${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}/setup.py
    @ONLY
    )

  # Add __init__.py and import from there all pybind11 functions and classes
  configure_file(
    ${PROJECT_SOURCE_DIR}/utilities/py4C/src/config/__init__.py.in
    ${PROJECT_BINARY_DIR}/${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}/__init__.py
    )

  # Add the py4C directory
  add_subdirectory(${PROJECT_SOURCE_DIR}/utilities/py4C)
endif()
