// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_SSI_ELCH_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_SSI_ELCH_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy_ssi.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
   * @brief Contact solving strategy with Nitsche's method.
   *
   * This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   * For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.
   */
  class NitscheStrategySsiElch : public NitscheStrategySsi
  {
   public:
    //! Shared data constructor
    NitscheStrategySsiElch(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        const Teuchos::ParameterList& params,
        std::vector<std::shared_ptr<CONTACT::Interface>> interface, int dim, const MPI_Comm& comm,
        double alphaf, int maxdof)
        : NitscheStrategySsi(data_ptr, dof_row_map, NodeRowMap, params, std::move(interface), dim,
              comm, alphaf, maxdof)
    {
    }

    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override
    {
      FOUR_C_THROW("not implemented");
    }

    void integrate(const CONTACT::ParamsInterface& cparams) override;

    //! don't want = operator
    NitscheStrategySsiElch operator=(const NitscheStrategySsiElch& old) = delete;

    //! don't want copy constructor
    NitscheStrategySsiElch(const NitscheStrategySsiElch& old) = delete;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
