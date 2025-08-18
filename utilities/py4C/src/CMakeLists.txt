# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# create python bindings module
pybind11_add_module(${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME} 4C_python_bindings_main.cpp)

set_target_properties(
  ${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}
  PROPERTIES PREFIX ""
             LIBRARY_OUTPUT_DIRECTORY
             "${PROJECT_BINARY_DIR}/${FOUR_C_PYTHON_BINDINGS_PROJECT_NAME}"
  )

add_subdirectory(config)
