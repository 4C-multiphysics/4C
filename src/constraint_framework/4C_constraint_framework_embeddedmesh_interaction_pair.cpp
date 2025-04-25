// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_cut_boundarycell.hpp"
#include "4C_cut_cutwizard.hpp"

#include <Teuchos_ENull.hpp>

FOUR_C_NAMESPACE_OPEN

Constraints::EMBEDDEDMESH::SolidInteractionPair::SolidInteractionPair(
    std::shared_ptr<Core::Elements::Element> element1, Core::Elements::Element* element2,
    Constraints::EMBEDDEDMESH::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard> cutwizard_ptr,
    std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells)
    : params_(params_ptr),
      element1_(element1),
      element2_(element2),
      cutwizard_ptr_(cutwizard_ptr),
      boundary_cells_(boundary_cells)
{
  // empty constructor
}

FOUR_C_NAMESPACE_CLOSE
