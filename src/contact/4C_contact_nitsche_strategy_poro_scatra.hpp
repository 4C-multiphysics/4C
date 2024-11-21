// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_PORO_SCATRA_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_PORO_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy_ssi.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class Coupling;
}

namespace CONTACT
{
  /*!
   \brief Contact solving strategy with Nitsche's method for poroscatra problem.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class NitscheStrategyPoroScatra : public NitscheStrategySsi
  {
   public:
    //! Standard constructor
    NitscheStrategyPoroScatra(const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<Teuchos::RCP<CONTACT::Interface>> interface,
        int dim, Teuchos::RCP<Epetra_Comm> comm, double alphaf, int maxdof);

    //! Shared data constructor
    NitscheStrategyPoroScatra(const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
        const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
        std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
        Teuchos::RCP<const Epetra_Comm> comm, double alphaf, int maxdof);

    void apply_force_stiff_cmt(Teuchos::RCP<Core::LinAlg::Vector<double>> dis,
        Teuchos::RCP<Core::LinAlg::SparseOperator>& kt,
        Teuchos::RCP<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override;

    void evaluate_reference_state() override;

    void integrate(const CONTACT::ParamsInterface& cparams) override;

    void set_state(
        const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) override;

    void set_parent_state(const enum Mortar::StateType& statename,
        const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis) override;

    Teuchos::RCP<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const enum CONTACT::VecBlockType& bp) const override;

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bp) const;

    // Flag for Poro No Penetration Condition
    bool has_poro_no_penetration() const override { return no_penetration_; }

    // don't want = operator and cctor
    NitscheStrategyPoroScatra operator=(const NitscheStrategyPoroScatra& old) = delete;
    NitscheStrategyPoroScatra(const NitscheStrategyPoroScatra& old) = delete;

   protected:
    // create an appropriate vector for the RHS
    Teuchos::RCP<Epetra_FEVector> setup_rhs_block_vec(
        const enum CONTACT::VecBlockType& bt) const override;

    // create an appropriate matrix block
    Teuchos::RCP<Core::LinAlg::SparseMatrix> setup_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bt) override;

    // complete matrix block with correct maps
    void complete_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bt, Teuchos::RCP<Core::LinAlg::SparseMatrix> kc) override;

    bool no_penetration_;

    //! Poro residual
    Teuchos::RCP<Epetra_FEVector> fp_;
    //! linearization of Poro residual w.r.t Poro dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kpp_;
    //! linearization of Poro residual w.r.t displacement dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kpd_;
    //! linearization of displacement residual w.r.t Poro dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kdp_;

    //! ScaTra residual
    Teuchos::RCP<Epetra_FEVector> fs_;
    //! linearization of ScaTra residual w.r.t ScaTra dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kss_;
    //! linearization of ScaTra residual w.r.t displacement dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> ksd_;
    //! linearization of displacement residual w.r.t ScaTra dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kds_;

    // It is assumed that there is no interaction between Poro and Scatra in terms of contact.
    // Therefore, kps_ and ksp_ do not exist.
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
