// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_constraintenforcer.hpp"

#include "4C_adapter_fld_fbi_movingboundary.hpp"
#include "4C_adapter_str_fbiwrapper.hpp"
#include "4C_beaminteraction_calc_utils.hpp"  // todo put this into bridge to keep everything beam specific in there
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_binstrategy.hpp"
#include "4C_fbi_adapter_constraintbridge.hpp"
#include "4C_fbi_adapter_constraintbridge_penalty.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_immersed_geometry_coupler.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_geometry_pair.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_rebalance_binning_based.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

Adapter::FBIConstraintenforcer::FBIConstraintenforcer(
    std::shared_ptr<Adapter::FBIConstraintBridge> bridge,
    std::shared_ptr<FBI::FBIGeometryCoupler> geometrycoupler)
    : fluid_(nullptr),
      structure_(nullptr),
      discretizations_(),
      bridge_(bridge),
      geometrycoupler_(geometrycoupler),
      column_structure_displacement_(nullptr),
      column_structure_velocity_(nullptr),
      column_fluid_velocity_(nullptr),
      velocity_pressure_splitter_(std::make_shared<Core::LinAlg::MapExtractor>())
{
}

/*----------------------------------------------------------------------*/

void Adapter::FBIConstraintenforcer::setup(std::shared_ptr<Adapter::FSIStructureWrapper> structure,
    std::shared_ptr<Adapter::FluidMovingBoundary> fluid)
{
  fluid_ = fluid;
  structure_ = structure;
  discretizations_.push_back(structure_->discretization());
  discretizations_.push_back(fluid_->discretization());

  Core::LinAlg::create_map_extractor_from_discretization(
      *(fluid_->discretization()), 3, *velocity_pressure_splitter_);

  const bool meshtying =
      (Global::Problem::instance()->fluid_dynamic_params().get<Inpar::FLUID::MeshTying>(
          "MESHTYING"));

  std::shared_ptr<Core::LinAlg::SparseOperator> fluidmatrix(nullptr);

  if (meshtying)
  {
    if (Core::Communication::num_mpi_ranks(structure_->discretization()->get_comm()) > 1)
      FOUR_C_THROW(
          "Currently fluid mesh tying can only be used for serial computations, since offproc "
          "assembly is not supported. Once the coupling matrices are computed by the fluid element "
          "owner, this will change.");

    fluidmatrix = (std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->get_meshtying())
                      ->init_system_matrix();
  }
  else
  {
    fluidmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(fluid_->discretization()->dof_row_map()), 30, true, true,
        Core::LinAlg::SparseMatrix::FE_MATRIX);  // todo Is there a better estimator?
  }

  bridge_->setup(structure_->discretization()->dof_row_map(),
      fluid_->discretization()->dof_row_map(), fluidmatrix, meshtying);
  if (Core::Communication::num_mpi_ranks(structure_->discretization()->get_comm()) > 1)
  {
    geometrycoupler_->extend_beam_ghosting(*(structure->discretization()));

    // After ghosting we need to explicitly set up the MultiMapExtractor again
    std::dynamic_pointer_cast<Adapter::FBIStructureWrapper>(structure_)
        ->setup_multi_map_extractor();
  }

  geometrycoupler_->setup(
      discretizations_, Core::Rebalance::get_col_version_of_row_vector(
                            *structure_->discretization(), structure_->dispnp()));
}

/*----------------------------------------------------------------------*/

void Adapter::FBIConstraintenforcer::evaluate()
{
  // We use the column vectors here, because currently the search is based on neighboring nodes,
  // but the element pairs are created using the elements needing all information on all their
  // DOFs
  column_structure_displacement_ = Core::Rebalance::get_col_version_of_row_vector(
      *structure_->discretization(), structure_->dispnp());
  column_structure_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
      *structure_->discretization(), structure_->velnp());
  column_fluid_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
      *fluid_->discretization(), std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->velnp());

  geometrycoupler_->update_binning(discretizations_[0], column_structure_displacement_);

  // Before each search we delete all pair and segment information
  bridge_->clear();
  bridge_->reset_bridge();

  // Do the search in the geometrycoupler_ and return the possible pair ids
  std::shared_ptr<std::map<int, std::vector<int>>> pairids =
      geometrycoupler_->search(discretizations_,
          column_structure_displacement_);  // todo make this a vector? At some point we probably
                                            // need the ale displacements as well

  // For now we need to separate the pair creation from the search, since the search takes place
  // on the fluid elements owner, while (for now) the pair has to be created on the beam element
  // owner
  create_pairs(pairids);

  // Create all needed matrix and vector contributions based on the current state
  bridge_->evaluate(discretizations_[0], discretizations_[1],
      std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->velnp(), structure_->velnp());
}

/*----------------------------------------------------------------------*/

std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FBIConstraintenforcer::structure_to_fluid(
    int step)
{
  // todo only access the parameter list once

  // Check if we want to couple the fluid
  const Teuchos::ParameterList& fbi = Global::Problem::instance()->fbi_params();
  if (Teuchos::getIntegralValue<FBI::BeamToFluidCoupling>(fbi, "COUPLING") !=
          FBI::BeamToFluidCoupling::solid &&
      fbi.get<int>("STARTSTEP") < step)
  {
    // Assemble the fluid stiffness matrix and hand it to the fluid solver
    std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->set_coupling_contributions(
        assemble_fluid_coupling_matrix());

    // Assemble the fluid force vector and hand it to the fluid solver
    fluid_->apply_interface_values(assemble_fluid_coupling_residual());
  }

  // return the current structure velocity
  return std::dynamic_pointer_cast<Adapter::FBIStructureWrapper>(structure_)
      ->extract_interface_velnp();
};

/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintenforcer::recompute_coupling_without_pair_creation()
{
  // Before each search we delete all pair and segment information
  bridge_->reset_bridge();

  reset_all_pair_states();

  // Create all needed matrix and vector contributions based on the current state
  bridge_->evaluate(discretizations_[0], discretizations_[1],
      std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->velnp(), structure_->velnp());
};

/*----------------------------------------------------------------------*/
// return the structure force
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FBIConstraintenforcer::fluid_to_structure()
{
  return assemble_structure_coupling_residual();
};

/*----------------------------------------------------------------------*/

// For now we need to separate the pair creation from the search, since the search takes place on
// the fluid elements owner, while (for now) the pair has to be created on the beam element owner
void Adapter::FBIConstraintenforcer::create_pairs(
    std::shared_ptr<std::map<int, std::vector<int>>> pairids)
{
  if (Core::Communication::num_mpi_ranks(structure_->discretization()->get_comm()) > 1)
  {
    // The geometrycoupler takes care of all MPI communication that needs to be done before the
    // pairs can finally be created
    geometrycoupler_->prepare_pair_creation(discretizations_, pairids);

    column_structure_displacement_ = Core::Rebalance::get_col_version_of_row_vector(
        *structure_->discretization(), structure_->dispnp());
    column_structure_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
        *structure_->discretization(), structure_->velnp());
    column_fluid_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
        *fluid_->discretization(), std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->velnp());
  }


  std::vector<Core::Elements::Element const*> ele_ptrs(2);
  std::vector<double> beam_dofvec = std::vector<double>();
  std::vector<double> fluid_dofvec = std::vector<double>();

  // loop over all (embedded) beam elements
  std::map<int, std::vector<int>>::const_iterator beamelementiterator;
  for (beamelementiterator = pairids->begin(); beamelementiterator != pairids->end();
      beamelementiterator++)
  {
    // add beam elements to the element pair pointer
    ele_ptrs[0] = (structure_->discretization())->g_element(beamelementiterator->first);


    if (ele_ptrs[0]->owner() !=
        Core::Communication::my_mpi_rank(structure_->discretization()->get_comm()))
      FOUR_C_THROW(
          "For now we can only create the pair on the beam owner, but beam element owner is %i "
          "and "
          "we are on proc %i \n",
          ele_ptrs[0]->owner(),
          Core::Communication::my_mpi_rank(structure_->discretization()->get_comm()));

    // loop over all fluid elements, in which the beam element might lie
    for (std::vector<int>::const_iterator fluideleIter = beamelementiterator->second.begin();
        fluideleIter != (beamelementiterator->second).end(); fluideleIter++)
    {
      Core::Elements::Element* fluidele = (fluid_->discretization())->g_element(*fluideleIter);

      // add fluid element to the element pair pointer
      ele_ptrs[1] = fluidele;

      // Extract current element dofs, i.e. positions and velocities
      extract_current_element_dofs(ele_ptrs, beam_dofvec, fluid_dofvec);

      // Finally tell the bridge to create the pair
      bridge_->create_pair(ele_ptrs, beam_dofvec, fluid_dofvec);
    }
  }
}
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintenforcer::reset_all_pair_states()
{
  // Get current state
  column_structure_displacement_ = Core::Rebalance::get_col_version_of_row_vector(
      *structure_->discretization(), structure_->dispnp());
  column_structure_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
      *structure_->discretization(), structure_->velnp());
  column_fluid_velocity_ = Core::Rebalance::get_col_version_of_row_vector(
      *fluid_->discretization(), std::dynamic_pointer_cast<Adapter::FBIFluidMB>(fluid_)->velnp());

  std::vector<Core::Elements::Element const*> ele_ptrs(2);
  std::vector<double> beam_dofvec = std::vector<double>();
  std::vector<double> fluid_dofvec = std::vector<double>();

  for (auto pairiterator = bridge_->get_pairs()->begin();
      pairiterator != bridge_->get_pairs()->end(); pairiterator++)
  {
    ele_ptrs[0] = (*pairiterator)->element1();
    ele_ptrs[1] = (*pairiterator)->element2();

    // Extract current element dofs, i.e. positions and velocities
    extract_current_element_dofs(ele_ptrs, beam_dofvec, fluid_dofvec);

    // Finally tell the bridge to create the pair
    bridge_->reset_pair(beam_dofvec, fluid_dofvec, *pairiterator);
  }
}
/*----------------------------------------------------------------------*/

void Adapter::FBIConstraintenforcer::extract_current_element_dofs(
    std::vector<Core::Elements::Element const*> elements, std::vector<double>& beam_dofvec,
    std::vector<double>& fluid_dofvec) const
{
  std::vector<double> vel_tmp;

  // extract the current position of the beam element from the displacement vector
  BeamInteraction::Utils::extract_pos_dof_vec_absolute_values(*(structure_->discretization()),
      elements[0], *column_structure_displacement_,
      beam_dofvec);  // todo get "interface" displacements only for beam
                     // elements
  // extract velocity of the beam element
  BeamInteraction::Utils::extract_pos_dof_vec_values(
      *(structure_->discretization()), elements[0], *column_structure_velocity_, vel_tmp);

  for (double val : vel_tmp) beam_dofvec.push_back(val);

  vel_tmp.clear();
  // extract the current positions and velocities of the fluid element todo only valid for fixed
  // grid, not for ALE
  fluid_dofvec.clear();
  const Core::Nodes::Node* const* fluidnodes = elements[1]->nodes();
  for (int lid = 0; lid < elements[1]->num_node(); ++lid)
  {
    for (int dim = 0; dim < 3; dim++)
    {
      fluid_dofvec.push_back(fluidnodes[lid]->x()[dim]);
    }
  }

  // extract current fluid velocities
  BeamInteraction::Utils::get_current_element_dis(
      *(fluid_->discretization()), elements[1], *column_fluid_velocity_, vel_tmp);

  // todo This is a very crude way to separate the pressure from the velocity dofs.. maybe just
  // use an extractor?
  for (unsigned int i = 0; i < vel_tmp.size(); i++)
  {
    if ((i + 1) % 4) fluid_dofvec.push_back(vel_tmp[i]);
  }
}

/*----------------------------------------------------------------------*/

void Adapter::FBIConstraintenforcer::set_binning(
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binning)
{
  geometrycoupler_->set_binning(binning);
};

FOUR_C_NAMESPACE_CLOSE
