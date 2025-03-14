// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_constraintenforcer_factory.hpp"

#include "4C_fbi_adapter_constraintbridge.hpp"
#include "4C_fbi_adapter_constraintbridge_penalty.hpp"
#include "4C_fbi_constraintenforcer_penalty.hpp"
#include "4C_fbi_immersed_geometry_coupler.hpp"
#include "4C_fbi_immersedcoupler_factory.hpp"
#include "4C_fbi_input.hpp"
#include "4C_inpar_fsi.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Adapter::FBIConstraintenforcer> Adapter::ConstraintEnforcerFactory::create_enforcer(
    const Teuchos::ParameterList& fsidyn, const Teuchos::ParameterList& fbidyn)
{
  std::shared_ptr<Adapter::FBIConstraintBridge> bridge =
      std::shared_ptr<Adapter::FBIConstraintBridgePenalty>(new Adapter::FBIConstraintBridgePenalty);

  std::shared_ptr<FBI::FBIGeometryCoupler> coupler =
      FBI::GeometryCouplerFactory::create_geometry_coupler(fbidyn);

  return std::shared_ptr<Adapter::FBIConstraintenforcer>(
      new Adapter::FBIPenaltyConstraintenforcer(bridge, coupler));
}

FOUR_C_NAMESPACE_CLOSE
