// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_AUGMENTED_LAGRANGE_STRATEGY_HPP
#define FOUR_C_CONTACT_MESHTYING_AUGMENTED_LAGRANGE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_meshtying_lagrange_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /**
   * Meshtying strategy using an augmented Lagrange saddle-point system.
   *
   * The displacement block is augmented by
   *
   *   penalty * B^T * W^{-1} * B,
   *
   * where B = [D, -M] and W contains the integrated interface weights.
   * The Lagrange multipliers remain independent unknowns.
   */
  class MtAugmentedLagrangeStrategy final : public MtLagrangeStrategy
  {
   public:
    MtAugmentedLagrangeStrategy(const Core::LinAlg::Map* dof_row_map,
        const Core::LinAlg::Map* node_row_map, Teuchos::ParameterList params,
        std::vector<std::shared_ptr<Mortar::Interface>> interfaces, int spatial_dim,
        const MPI_Comm& comm, double alphaf, int maxdof);

    void mortar_coupling(const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis) override;

   private:
    //! Inverse augmentation scaling matrix penalty * W^{-1}
    std::shared_ptr<Core::LinAlg::Vector<double>> augmentation_scaling_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif