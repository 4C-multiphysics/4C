// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_str_model_evaluator_partitioned.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_ssi_partitioned.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::PartitionedSSI::PartitionedSSI(
    const std::shared_ptr<const SSI::SSIPart> ssi_part)
    : ssi_part_(ssi_part)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedSSI::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  // perform structural meshtying
  if (ssi_part_->ssi_interface_meshtying())
  {
    // cast old Jacobian
    auto& jac_sparse = dynamic_cast<Core::LinAlg::SparseMatrix&>(jac);

    auto map_structure_interior = ssi_part_->ssi_structure_mesh_tying()->interior_map();
    auto cond_master_dof_map = ssi_part_->ssi_structure_mesh_tying()->full_master_side_map();

    // initialize new Jacobian
    Core::LinAlg::SparseMatrix jac_new(*global_state().dof_row_map(), 81, true, true);

    // assemble interior rows and columns of original Jacobian into new Jacobian
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *map_structure_interior,
        *map_structure_interior, 1.0, nullptr, nullptr, jac_new, true, true);

    // assemble interior rows and master-side columns of original Jacobian into new Jacobian
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *map_structure_interior,
        *cond_master_dof_map, 1.0, nullptr, nullptr, jac_new, true, true);

    // assemble master-side rows and interior columns of original Jacobian into new Jacobian
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_master_dof_map,
        *map_structure_interior, 1.0, nullptr, nullptr, jac_new, true, true);

    // assemble master-side rows and columns of original Jacobian into new Jacobian
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_master_dof_map,
        *cond_master_dof_map, 1.0, nullptr, nullptr, jac_new, true, true);

    for (const auto& meshtying : ssi_part_->ssi_structure_mesh_tying()->mesh_tying_handlers())
    {
      auto cond_slave_dof_map = meshtying->slave_master_coupling()->source_dof_map();
      auto converter = meshtying->slave_side_converter();

      // transform and assemble slave-side rows of original Jacobian into new Jacobian (interior
      // columns)
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_slave_dof_map,
          *map_structure_interior, 1.0, &(*converter), nullptr, jac_new, true, true);

      // transform and assemble slave-side rows of original Jacobian into new Jacobian (master-side
      // columns)
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_slave_dof_map,
          *cond_master_dof_map, 1.0, &(*converter), nullptr, jac_new, true, true);

      // transform and assemble slave-side columns of original Jacobian into new Jacobian (interior
      // rows)
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *map_structure_interior,
          *cond_slave_dof_map, 1.0, nullptr, &(*converter), jac_new, true, true);

      // transform and assemble slave-side columns of original Jacobian into new Jacobian
      // (master-side rows)
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_master_dof_map,
          *cond_slave_dof_map, 1.0, nullptr, &(*converter), jac_new, true, true);

      for (const auto& meshtying2 : ssi_part_->ssi_structure_mesh_tying()->mesh_tying_handlers())
      {
        auto cond_slave_dof_map2 = meshtying2->slave_master_coupling()->source_dof_map();
        auto converter2 = meshtying2->slave_side_converter();

        // assemble derivatives of surface slave dofs w.r.t. line slave dofs (block l)
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(jac_sparse, *cond_slave_dof_map,
            *cond_slave_dof_map2, 1.0, &(*converter), &(*converter2), jac_new, true, true);
      }
    }

    auto slavemaps = ssi_part_->ssi_structure_mesh_tying()->full_slave_side_map();

    // subject slave-side rows of new Jacobian to pseudo Dirichlet conditions to finalize
    // structural meshtying
    jac_new.complete();
    jac_new.apply_dirichlet(*slavemaps);

    // replace old Jacobian by new one
    jac_sparse.assign(Core::LinAlg::DataAccess::Share, jac_new);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedSSI::run_pre_compute_x(
    const Core::LinAlg::Vector<double>& xold, Core::LinAlg::Vector<double>& dir_mutable,
    const NOX::Nln::Group& curr_grp)
{
  // perform structural meshtying
  if (ssi_part_->ssi_interface_meshtying())
  {
    for (const auto& meshtying : ssi_part_->ssi_structure_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_map_extractor = meshtying->slave_master_extractor();

      // transform and assemble master-side part of structural increment vector to slave side
      coupling_map_extractor->insert_vector(
          *meshtying->slave_master_coupling()->target_to_source(
              *coupling_map_extractor->extract_vector(dir_mutable, 2)),
          1, dir_mutable);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedSSI::setup()
{
  check_init();

  Solid::ModelEvaluator::BaseSSI::setup();

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedSSI::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  // perform structural meshtying
  if (ssi_part_->ssi_interface_meshtying() and ssi_part_->is_setup())
  {
    for (const auto& meshtying : ssi_part_->ssi_structure_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_map_extractor = meshtying->slave_master_extractor();
      // transform and assemble slave-side part of structural right-hand side vector to master side
      coupling_map_extractor->add_vector(*meshtying->slave_master_coupling()->source_to_target(
                                             *coupling_map_extractor->extract_vector(f, 1)),
          2, f);

      // zero out slave-side part of structural right-hand side vector
      coupling_map_extractor->put_scalar(f, 1, 0.0);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
