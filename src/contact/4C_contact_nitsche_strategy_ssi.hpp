// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_SSI_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_SSI_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
   * @brief Contact solving strategy with Nitsche's method.
   *
   * This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   * For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.
   */
  class NitscheStrategySsi : public NitscheStrategy
  {
   public:
    //! Standard constructor
    NitscheStrategySsi(const Core::LinAlg::Map* dofRowMap, const Core::LinAlg::Map* nodeRowMap,
        const Teuchos::ParameterList& params,
        std::vector<std::shared_ptr<CONTACT::Interface>> interface, int dim, const MPI_Comm& comm,
        double alphaf, int maxdof)
        : NitscheStrategy(
              dofRowMap, nodeRowMap, params, std::move(interface), dim, comm, alphaf, maxdof)
    {
    }

    //! Shared data constructor
    NitscheStrategySsi(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dofRowMap, const Core::LinAlg::Map* nodeRowMap,
        const Teuchos::ParameterList& params,
        std::vector<std::shared_ptr<CONTACT::Interface>> interface, int dim, const MPI_Comm& comm,
        double alphaf, int maxdof)
        : NitscheStrategy(data_ptr, dofRowMap, nodeRowMap, params, std::move(interface), dim, comm,
              alphaf, maxdof)
    {
    }

    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override
    {
      if (kt != nullptr && f != nullptr)
      {
        NitscheStrategy::apply_force_stiff_cmt(dis, kt, f, step, iter, predictor);
      }
      curr_state_eval_ = true;
    }

    void evaluate_reference_state() override;

    void integrate(const CONTACT::ParamsInterface& cparams) override;

    void set_state(
        const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) override;

    void set_parent_state(const enum Mortar::StateType& statename,
        const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis) override;

    std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const enum CONTACT::VecBlockType& bp) const override;

    /*!
     * @brief get the pointer to the matrix block
     *
     * @param[in] bt   block type of requested matrix block
     * @return pointer to matrix block
     */
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bp) const;

    //! don't want = operator
    NitscheStrategySsi operator=(const NitscheStrategySsi& old) = delete;
    //! don't want copy constructor
    NitscheStrategySsi(const NitscheStrategySsi& old) = delete;

   protected:
    std::shared_ptr<Core::LinAlg::FEVector<double>> setup_rhs_block_vec(
        const enum CONTACT::VecBlockType& bt) const override;

    std::shared_ptr<Core::LinAlg::SparseMatrix> setup_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bt) override;

    void complete_matrix_block_ptr(const enum CONTACT::MatBlockType& bt,
        std::shared_ptr<Core::LinAlg::SparseMatrix> kc) override;

    //! current scalar state vector
    std::shared_ptr<Core::LinAlg::Vector<double>> curr_state_scalar_;
    //! ScaTra residual
    std::shared_ptr<Core::LinAlg::FEVector<double>> fs_;
    //! linearization of ScaTra residual w.r.t ScaTra dofs
    std::shared_ptr<Core::LinAlg::SparseMatrix> kss_;
    //! linearization of ScaTra residual w.r.t displacement dofs
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksd_;
    //! linearization of displacement residual w.r.t ScaTra dofs
    std::shared_ptr<Core::LinAlg::SparseMatrix> kds_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
