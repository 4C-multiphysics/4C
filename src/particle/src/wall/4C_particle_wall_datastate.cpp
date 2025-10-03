// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_wall_datastate.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_particle_input.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEWALL::WallDataState::WallDataState(const Teuchos::ParameterList& params) : params_(params)
{
  // empty constructor
}

void PARTICLEWALL::WallDataState::init(
    const std::shared_ptr<Core::FE::Discretization> walldiscretization)
{
  // set wall discretization
  walldiscretization_ = walldiscretization;

  // get flags defining considered states of particle wall
  const bool ismoving = params_.get<bool>("PARTICLE_WALL_MOVING");
  const bool isloaded = params_.get<bool>("PARTICLE_WALL_LOADED");

  // set current dof row and column map
  curr_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*walldiscretization_->dof_row_map());

  // create states needed for moving walls
  if (ismoving)
  {
    disp_row_ = std::make_shared<Core::LinAlg::Vector<double>>(*curr_dof_row_map_);
    disp_col_ = std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map());
    disp_row_last_transfer_ = std::make_shared<Core::LinAlg::Vector<double>>(*curr_dof_row_map_);
    vel_col_ = std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map());
    acc_col_ = std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map());
  }

  // create states needed for loaded walls
  if (isloaded)
  {
    force_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map());
  }
}

void PARTICLEWALL::WallDataState::setup()
{
  // nothing to do
}

void PARTICLEWALL::WallDataState::check_for_correct_maps()
{
  if (disp_row_ != nullptr)
    if (not disp_row_->get_map().same_as(*walldiscretization_->dof_row_map()))
      FOUR_C_THROW("map of state 'disp_row_' corrupt!");

  if (disp_col_ != nullptr)
    if (not disp_col_->get_map().same_as(*walldiscretization_->dof_col_map()))
      FOUR_C_THROW("map of state 'disp_col_' corrupt!");

  if (disp_row_last_transfer_ != nullptr)
    if (not disp_row_last_transfer_->get_map().same_as(*walldiscretization_->dof_row_map()))
      FOUR_C_THROW("map of state 'disp_row_last_transfer_' corrupt!");

  if (vel_col_ != nullptr)
    if (not vel_col_->get_map().same_as(*walldiscretization_->dof_col_map()))
      FOUR_C_THROW("map of state 'vel_col_' corrupt!");

  if (acc_col_ != nullptr)
    if (not acc_col_->get_map().same_as(*walldiscretization_->dof_col_map()))
      FOUR_C_THROW("map of state 'acc_col_' corrupt!");

  if (force_col_ != nullptr)
    if (not force_col_->get_map().same_as(*walldiscretization_->dof_col_map()))
      FOUR_C_THROW("map of state 'force_col_' corrupt!");
}

void PARTICLEWALL::WallDataState::update_maps_of_state_vectors()
{
  if (disp_row_ != nullptr and disp_col_ != nullptr)
  {
    // export row map based displacement vector
    std::shared_ptr<Core::LinAlg::Vector<double>> temp = disp_row_;
    disp_row_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_row_map(), true);
    Core::LinAlg::export_to(*temp, *disp_row_);

    // update column map based displacement vector
    disp_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map(), true);
    Core::LinAlg::export_to(*disp_row_, *disp_col_);

    // store displacements after last transfer
    disp_row_last_transfer_ = std::make_shared<Core::LinAlg::Vector<double>>(*disp_row_);
  }

  if (vel_col_ != nullptr)
  {
    // export old column to old row map based vector (no communication)
    Core::LinAlg::Vector<double> temp(*curr_dof_row_map_);
    Core::LinAlg::export_to(*vel_col_, temp);
    // export old row map based vector to new column map based vector
    vel_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map(), true);
    Core::LinAlg::export_to(temp, *vel_col_);
  }

  if (acc_col_ != nullptr)
  {
    // export old column to old row map based vector (no communication)
    Core::LinAlg::Vector<double> temp(*curr_dof_row_map_);
    Core::LinAlg::export_to(*acc_col_, temp);
    // export old row map based vector to new column map based vector
    acc_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map(), true);
    Core::LinAlg::export_to(temp, *acc_col_);
  }

  if (force_col_ != nullptr)
  {
    // export old column to old row map based vector (no communication)
    Core::LinAlg::Vector<double> temp(*curr_dof_row_map_);
    Core::LinAlg::export_to(*force_col_, temp);
    // export old row map based vector to new column map based vector
    force_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*walldiscretization_->dof_col_map(), true);
    Core::LinAlg::export_to(temp, *force_col_);
  }

  // set new dof row map
  curr_dof_row_map_ = std::make_shared<Core::LinAlg::Map>(*walldiscretization_->dof_row_map());
}

FOUR_C_NAMESPACE_CLOSE
