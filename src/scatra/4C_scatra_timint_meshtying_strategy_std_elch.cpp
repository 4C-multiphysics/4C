// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_meshtying_strategy_std_elch.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_utils_splitstrategy.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               fang 12/14 |
 *----------------------------------------------------------------------*/
ScaTra::MeshtyingStrategyStdElch::MeshtyingStrategyStdElch(ScaTra::ScaTraTimIntElch* elchtimint)
    : MeshtyingStrategyStd(elchtimint)
{
}  // ScaTra::MeshtyingStrategyStdElch::MeshtyingStrategyStdElch


/*----------------------------------------------------------------------*
 | initialize system matrix for electrochemistry problems    fang 12/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator> ScaTra::MeshtyingStrategyStdElch::init_system_matrix()
    const
{
  std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix;

  // initialize standard (stabilized) system matrix (and save its graph)
  switch (scatratimint_->matrix_type())
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      systemmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *scatratimint_->discretization()->dof_row_map(), 27, false, true);
      break;
    }

    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      systemmatrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatratimint_->dof_block_maps(), *scatratimint_->dof_block_maps(), 81, false, true);
      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown matrix type of ScaTra field");
    }
  }

  return systemmatrix;
}  // ScaTra::MeshtyingStrategyStdElch::init_system_matrix


/*------------------------------------------------------------------------*
 | instantiate strategy for Newton-Raphson convergence check   fang 02/16 |
 *------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyStdElch::init_conv_check_strategy()
{
  if (elch_tim_int()->macro_scale())
  {
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyStdMacroScaleElch>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
  }
  else
  {
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyStdElch>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
  }
}  // ScaTra::MeshtyingStrategyStdElch::init_conv_check_strategy

FOUR_C_NAMESPACE_CLOSE
