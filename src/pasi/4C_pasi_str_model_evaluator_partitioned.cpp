// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_str_model_evaluator_partitioned.hpp"

#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_implicit.hpp"


FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Solid::ModelEvaluator::PartitionedPASI::PartitionedPASI()
{
  // empty constructor
}

void Solid::ModelEvaluator::PartitionedPASI::setup()
{
  // pasi interface force at t_{n+1}
  interface_force_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);

  // set flag
  issetup_ = true;
}

std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::PartitionedPASI::get_block_dof_row_map_ptr() const
{
  check_init_setup();
  return global_state().dof_row_map();
}

std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::PartitionedPASI::get_current_solution_ptr() const
{
  check_init();
  return global_state().get_dis_np();
}

std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::PartitionedPASI::get_last_time_step_solution_ptr() const
{
  check_init();
  return global_state().get_dis_n();
}

void Solid::ModelEvaluator::PartitionedPASI::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  Core::LinAlg::assemble_my_vector(1.0, f, -timefac_np, *interface_force_np_ptr_);
}

void Solid::ModelEvaluator::PartitionedPASI::update_step_state(const double& timefac_n) { return; }

FOUR_C_NAMESPACE_CLOSE
