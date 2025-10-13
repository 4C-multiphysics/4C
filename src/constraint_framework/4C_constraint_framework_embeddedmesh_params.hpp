// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_PARAMS_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_input.hpp"
#include "4C_cut_enum.hpp"
#include "4C_fem_general_cell_type.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Constraints::EmbeddedMesh
{
  struct EmbeddedMeshParams
  {
    //! Strategy for coupling the embedded meshes
    CouplingStrategy coupling_strategy_;

    //! Constraint enforcement method
    EnforcementStrategy constraint_enforcement_;

    //! Penalty parameter for coupling enforcement
    double constraint_penalty_parameter_ = 0.0;

    //! Shape function for the mortar Lagrange-multiplicators
    SolidToSolidMortarShapefunctions mortar_shape_function_;

    //! Stabilization parameter for coupling with the Nitsche method
    double nitsche_stabilization_param_ = 0.0;

    //! Average weight gamma for coupling with the Nitsche method
    double nitsche_average_weight_gamma_ = 0.0;

    //! Nodal Dof set strategy for XFEM
    Cut::NodalDofSetStrategy xfem_nodal_dof_set_strategy_ =
        Cut::NodalDofSetStrategy::NDS_Strategy_OneDofset_PerNodeAndPosition;

    //! Integration method for volume cells for XFEM
    Cut::VCellGaussPts xfem_volume_cell_gauss_point_by_ =
        Cut::VCellGaussPts::VCellGaussPts_Tessellation;

    //! Integration method for boundary cells for XFEM
    Cut::BCellGaussPts xfem_bcell_gauss_point_by_ = Cut::BCellGaussPts::BCellGaussPts_Tessellation;

    //! Get gmsh output of cut
    bool gmsh_cut_out_ = false;

    //! Print coutput of cut on the screen
    bool cut_screen_output_ = false;

    //! Parameter list of cut
    Teuchos::ParameterList cut_params_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif