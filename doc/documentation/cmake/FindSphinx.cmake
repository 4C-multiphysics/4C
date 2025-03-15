# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

include(FindPackageHandleStandardArgs)

## We are likely to find Sphinx near the Python interpreter
find_program(
  SPHINX_EXECUTABLE
  NAMES "${FOUR_C_PYTHON_VENV_BUILD}/bin/sphinx-multibuild"
  DOC "Sphinx documentation generator"
  )

mark_as_advanced(SPHINX_EXECUTABLE)

find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)
