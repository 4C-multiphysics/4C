# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_module()

set(_dependencies
    # cmake-format: sortable
    adapter
    beam3
    beaminteraction
    config
    contact
    core
    global_data
    inpar
    solver_nonlin_nox
    timestepping
    )

four_c_add_dependency(${AUTO_DEFINED_MODULE_NAME} ${_dependencies})

four_c_unity_build_compile_separately(
  ${AUTO_DEFINED_MODULE_NAME} 4C_structure_new_timint_basedataglobalstate.cpp
  )
