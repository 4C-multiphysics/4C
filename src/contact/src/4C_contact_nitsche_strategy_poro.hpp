// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_PORO_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_PORO_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief Contact solving strategy with Nitsche's method.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class NitscheStrategyPoro : public NitscheStrategy
  {
   public:
    //! Standard constructor
    NitscheStrategyPoro(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        int dim, MPI_Comm comm, double alphaf, int maxdof)
        : NitscheStrategy(
              dof_row_map, NodeRowMap, params, std::move(interface), dim, comm, alphaf, maxdof),
          no_penetration_(params.get<bool>("CONTACT_NO_PENETRATION"))
    {
    }

    //! Shared data constructor
    NitscheStrategyPoro(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        int dim, MPI_Comm comm, double alphaf, int maxdof)
        : NitscheStrategy(data_ptr, dof_row_map, NodeRowMap, params, std::move(interface), dim,
              comm, alphaf, maxdof),
          no_penetration_(params.get<bool>("CONTACT_NO_PENETRATION"))
    {
    }

    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override;

    void integrate(const CONTACT::ParamsInterface& cparams) override;

    void set_state(
        const Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) override;

    void set_parent_state(const Mortar::StateType& statename,
        const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis) override;

    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const CONTACT::VecBlockType& bp) const override;

    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const CONTACT::MatBlockType& bp) const;


    // Flag for Poro No Penetration Condition
    bool has_poro_no_penetration() const override { return no_penetration_; }

    // don't want = operator and cctor
    NitscheStrategyPoro operator=(const NitscheStrategyPoro& old) = delete;
    NitscheStrategyPoro(const NitscheStrategyPoro& old) = delete;

   protected:
    // create an appropriate vector for the RHS
    std::shared_ptr<Core::LinAlg::FEVector<double>> setup_rhs_block_vec(
        const CONTACT::VecBlockType& bt) const override;

    // create an appropriate matrix block
    std::shared_ptr<Core::LinAlg::SparseMatrix> setup_matrix_block_ptr(
        const CONTACT::MatBlockType& bt) override;

    // complete matrix block with correct maps
    void complete_matrix_block_ptr(
        const CONTACT::MatBlockType& bt, std::shared_ptr<Core::LinAlg::SparseMatrix> kc) override;

    bool no_penetration_;

    std::shared_ptr<Core::LinAlg::FEVector<double>> fp_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> kpp_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> kpd_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> kdp_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
