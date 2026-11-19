// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_meshtying_augmented_lagrange_strategy.hpp"

#include "4C_contact_input.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CONTACT::MtAugmentedLagrangeStrategy::MtAugmentedLagrangeStrategy(
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* node_row_map,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<Mortar::Interface>> interfaces,
    const int spatial_dim, const MPI_Comm& comm, const double alphaf, const int maxdof)
    : MtLagrangeStrategy(dof_row_map, node_row_map, std::move(params), std::move(interfaces),
          spatial_dim, comm, alphaf, maxdof)
{
  FOUR_C_ASSERT_ALWAYS(system_type() == CONTACT::SystemType::saddlepoint,
      "MtAugmentedLagrangeStrategy requires SYSTEM SaddlePoint.");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::MtAugmentedLagrangeStrategy::mortar_coupling(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis)
{
  // Assemble D, M, the multiplier maps, interface weights, etc.
  MtLagrangeStrategy::mortar_coupling(dis);

  const auto interface_weights = interface_scaling_vector();
  FOUR_C_ASSERT_ALWAYS(
      interface_weights != nullptr, "The meshtying interface weights are not available.");

  augmentation_scaling_ =
      std::make_shared<Core::LinAlg::Vector<double>>(interface_weights->get_map(), true);

  const double penalty_param = params().get<double>("PENALTYPARAM");
  FOUR_C_ASSERT_ALWAYS(
      penalty_param > 0.0, "The penalty parameter must be a finite positive value.");

  for (int lid = 0; lid < interface_weights->local_length(); ++lid)
  {
    const double interface_weight = interface_weights->get_values()[lid];

    if (interface_weight == 0.0)
    {
      augmentation_scaling_->replace_local_value(lid, 0.0);
    }
    else
    {
      FOUR_C_ASSERT_ALWAYS(std::isfinite(interface_weight) && interface_weight > 0.0,
          "Invalid interface scaling value {} at local index {}.", interface_weight, lid);

      augmentation_scaling_->replace_local_value(lid, penalty_param / interface_weight);
    }
  }

  // K_aug = penalty * [D^T; -M^T] * W^{-1} * [D, -M].
  // penalty * W^{-1} * D
  auto scaled_d = std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  scaled_d->left_scale(*augmentation_scaling_);

  // penalty * W^{-1} * M
  auto scaled_m = std::make_shared<Core::LinAlg::SparseMatrix>(*mmatrix_);
  scaled_m->left_scale(*augmentation_scaling_);

  // Products already contain penalty * W^{-1}.
  std::shared_ptr<Core::LinAlg::SparseMatrix> dtd =
      Core::LinAlg::matrix_multiply(*dmatrix_, true, *scaled_d, false, false, false, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> dtm =
      Core::LinAlg::matrix_multiply(*dmatrix_, true, *scaled_m, false, false, false, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> mtd =
      Core::LinAlg::matrix_multiply(*mmatrix_, true, *scaled_d, false, false, false, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> mtm =
      Core::LinAlg::matrix_multiply(*mmatrix_, true, *scaled_m, false, false, false, true);

  if (par_redist())
  {
    dtd = Core::LinAlg::matrix_row_transform(*dtd, *non_redist_gsdofrowmap_);
    dtm = Core::LinAlg::matrix_row_transform(*dtm, *non_redist_gsdofrowmap_);
    mtd = Core::LinAlg::matrix_row_transform(*mtd, *non_redist_gtdofrowmap_);
    mtm = Core::LinAlg::matrix_row_transform(*mtm, *non_redist_gtdofrowmap_);
  }

  augmentation_stiffness_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);

  Core::LinAlg::matrix_add(*dtd, false, 1.0, *augmentation_stiffness_, 1.0);
  Core::LinAlg::matrix_add(*dtm, false, -1.0, *augmentation_stiffness_, 1.0);
  Core::LinAlg::matrix_add(*mtd, false, -1.0, *augmentation_stiffness_, 1.0);
  Core::LinAlg::matrix_add(*mtm, false, 1.0, *augmentation_stiffness_, 1.0);

  augmentation_stiffness_->complete();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::MtAugmentedLagrangeStrategy::evaluate_force(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // Adds B^T * lambda.
  MtLagrangeStrategy::evaluate_force(dis);

  FOUR_C_ASSERT_ALWAYS(
      augmentation_stiffness_ != nullptr, "Augmentation stiffness is not available.");

  Core::LinAlg::Vector<double> augmentation_force(*problem_dofs(), true);
  augmentation_stiffness_->multiply(false, *dis, augmentation_force);

  // Adds penalty * B^T * W^{-1} * B * u.
  f_->update(1.0, augmentation_force, 1.0);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::MtAugmentedLagrangeStrategy::evaluate_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // Builds B^T, B and the zero multiplier block.
  MtLagrangeStrategy::evaluate_stiff(dis);

  FOUR_C_ASSERT_ALWAYS(
      augmentation_stiffness_ != nullptr, "Augmentation stiffness is not available.");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::MtAugmentedLagrangeStrategy::evaluate_force_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  evaluate_force(dis);
  evaluate_stiff(dis);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
CONTACT::MtAugmentedLagrangeStrategy::get_matrix_block_ptr(
    const CONTACT::MatBlockType& block_type) const
{
  if (block_type == CONTACT::MatBlockType::displ_displ)
  {
    FOUR_C_ASSERT_ALWAYS(
        augmentation_stiffness_ != nullptr, "Augmentation stiffness is not available.");

    return augmentation_stiffness_;
  }

  return MtLagrangeStrategy::get_matrix_block_ptr(block_type);
}

FOUR_C_NAMESPACE_CLOSE