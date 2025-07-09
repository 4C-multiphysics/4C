// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::BeamInteractionDataState::BeamInteractionDataState()
    : isinit_(false),
      issetup_(false),
      myrank_(0),
      dis_(nullptr),
      dis_restart_(nullptr),
      dis_restart_col_(nullptr),
      is_restart_coupling_(false),
      disnp_(nullptr),
      discolnp_(nullptr),
      forcen_(nullptr),
      forcenp_(nullptr),
      stiff_(nullptr)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteractionDataState::init()
{
  // We have to call setup() after init()
  issetup_ = false;

  // clear stl stuff
  bintorowelemap_.clear();
  exbintorowelemap_.clear();
  roweletobinmap_.clear();

  // end of initialization
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BeamInteractionDataState::setup(
    std::shared_ptr<const Core::FE::Discretization> const& ia_discret)
{
  // safety check
  check_init();

  myrank_ = Core::Communication::my_mpi_rank(ia_discret->get_comm());

  // displacements
  dis_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, ia_discret->dof_row_map(), true);
  disnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret->dof_col_map());
  discolnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*ia_discret->dof_col_map());

  // force
  forcen_ = std::make_shared<Core::LinAlg::FEVector<double>>(
      ia_discret->dof_row_map()->get_epetra_block_map());
  forcenp_ = std::make_shared<Core::LinAlg::FEVector<double>>(
      ia_discret->dof_row_map()->get_epetra_block_map());

  stiff_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *ia_discret->dof_row_map(), 81, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
