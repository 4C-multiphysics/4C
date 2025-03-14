// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_adapter_constraintbridge.hpp"

#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_pair_factory.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_fluid_assembly_strategy.hpp"
#include "4C_fbi_fluidblockmatrix_assembly_strategy.hpp"
#include "4C_fbi_input.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_sparseoperator.hpp"

FOUR_C_NAMESPACE_OPEN

Adapter::FBIConstraintBridge::FBIConstraintBridge()
    : beam_interaction_params_(nullptr),
      assemblystrategy_(nullptr),
      meshtying_pairs_(
          std::make_shared<std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>>()),
      geometry_evaluation_data_(nullptr) {};
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridge::setup(const Epetra_Map* beam_map, const Epetra_Map* fluid_map,
    std::shared_ptr<Core::LinAlg::SparseOperator> fluidmatrix, bool fluidmeshtying)
{
  // Create the beaminteraction data container and set the parameters
  beam_interaction_params_ = std::make_shared<FBI::BeamToFluidMeshtyingParams>();
  beam_interaction_params_->init();
  beam_interaction_params_->setup();

  const Teuchos::ParameterList& geometry_parameter_list =
      Global::Problem::instance()->fbi_params().sublist("BEAM TO FLUID MESHTYING");

  // Create the beaminteraction data container and set the parameters
  geometry_evaluation_data_ =
      std::make_shared<GEOMETRYPAIR::LineTo3DEvaluationData>(geometry_parameter_list);

  if (fluidmeshtying)
  {
    // For the option condensed smat this can be changed by creating a FEMatrix instead of a
    // CRSMatrix!
    if (beam_interaction_params_->get_contact_discretization() ==
        FBI::BeamToFluidDiscretization::mortar)
      FOUR_C_THROW("Fluid Meshtying is not supported when using a mortar discretization!");

    assemblystrategy_ = std::make_shared<FBI::Utils::FBIBlockAssemblyStrategy>();
  }
  else
    assemblystrategy_ = std::make_shared<FBI::Utils::FBIAssemblyStrategy>();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridge::create_pair(
    const std::vector<Core::Elements::Element const*> elements,
    const std::vector<double> beam_centerline_dofvec, const std::vector<double> fluid_nodal_dofvec)
{
  // create a new beaminteratcion pair
  std::shared_ptr<BeamInteraction::BeamContactPair> newinteractionpair =
      FBI::PairFactory::create_pair(elements, *get_params());

  // create the underlying geometrypair doing the integration (segment or gauss point projection
  // based)
  newinteractionpair->create_geometry_pair(elements[0], elements[1], get_geometry_data());
  newinteractionpair->init(get_params(), elements);
  newinteractionpair->setup();

  // hand in the current position and velocities of the participating elements
  reset_pair(beam_centerline_dofvec, fluid_nodal_dofvec, newinteractionpair);

  // add to list of current contact pairs
  meshtying_pairs_->push_back(newinteractionpair);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridge::reset_pair(const std::vector<double> beam_centerline_dofvec,
    const std::vector<double> fluid_nodal_dofvec,
    std::shared_ptr<BeamInteraction::BeamContactPair> interactionpair)
{
  // hand in the current position and velocities of the participating elements
  interactionpair->reset_state(beam_centerline_dofvec, fluid_nodal_dofvec);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridge::clear()
{
  // Delete all pairs and segmentation information
  meshtying_pairs_->clear();
  geometry_evaluation_data_->clear();
}

FOUR_C_NAMESPACE_CLOSE
