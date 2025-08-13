# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_module()

set(_dependencies
    # cmake-format: sortable
    ale
    art_net
    beam3
    beaminteraction
    bele
    config
    constraint
    contact
    core
    fluid
    fluid_ele
    fluid_xfluid
    lubrication
    mat
    membrane
    module_registry
    mortar
    particle_engine
    porofluid_pressure_based_elast_scatra
    porofluid_pressure_based_ele
    red_airways
    reduced_lung
    rigidsphere
    scatra_ele
    shell7p
    shell_kl_nurbs
    solid_3D_ele
    solid_poro_3D_ele
    solid_scatra_3D_ele
    structure_new
    thermo
    torsion3
    truss3
    w1
    )

four_c_add_internal_dependency(${AUTO_DEFINED_MODULE_NAME} ${_dependencies})
