# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Build the common test functionality
add_subdirectory(common)

# List all test directories here
add_subdirectory(beam3)
add_subdirectory(beaminteraction)
add_subdirectory(contact)
add_subdirectory(fbi)
add_subdirectory(geometry_pair)
add_subdirectory(io)
add_subdirectory(mat)
add_subdirectory(mixture)
add_subdirectory(mortar)
add_subdirectory(particle_engine)
add_subdirectory(particle_interaction)
add_subdirectory(particle_rigidbody)
add_subdirectory(porofluid_pressure_based_elast_scatra)
add_subdirectory(scatra)
add_subdirectory(solid_3D_ele)
