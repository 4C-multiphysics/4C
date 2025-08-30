// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_TEKO_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_TEKO_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <MueLu_UseDefaultTypes.hpp>
#include <Teko_LU2x2Strategy.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /*! \brief Set of standard block-matrix preconditioners

    Teko only needs the parameters for the block inverses in the parameter list
    as sublists "Inverse1", "Inverse2",...
    From the parameter list the Teko lists are automatically constructed, no 4C
    SOLVER objects needed!
   */
  class TekoPreconditioner : public PreconditionerTypeBase
  {
   public:
    TekoPreconditioner(Teuchos::ParameterList& tekolist);

    void setup(Core::LinAlg::SparseOperator& matrix, const Core::LinAlg::MultiVector<double>& x,
        Core::LinAlg::MultiVector<double>& b) override;

    /// linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const override { return p_; }

   private:
    Teuchos::ParameterList& tekolist_;

    //! system of equations used for preconditioning used by P_ only
    Teuchos::RCP<const Thyra::LinearOpBase<double>> pmatrix_;

    //! preconditioner
    std::shared_ptr<Epetra_Operator> p_;
  };


  /**
   * \brief A special routine for a block LU2x2 method using a sparse approximate inverse
   *
   * This special block LU2x2 routine uses a sparse approximate inverse to compute an explicit
   * Schur complement. Further information on the implementation can be found in:
   * M. Firmbach, I. Steinbrecher, A. Popp and M. Mayr: An approximate block factorization
   * preconditioner for mixed-dimensional beam-solid interaction.
   * Computer Methods in Applied Mechanics and Engineering, 431:838-853, 2024,
   * https://doi.org/10.1016/j.cma.2024.117256
   */
  class LU2x2SpaiStrategy : public Teko::LU2x2Strategy
  {
   public:
    LU2x2SpaiStrategy() = default;

    LU2x2SpaiStrategy(const Teuchos::RCP<Teko::InverseFactory>& invFA,
        const Teuchos::RCP<Teko::InverseFactory>& invS);

    const Teko::LinearOp getHatInvA00(
        const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const override;

    const Teko::LinearOp getTildeInvA00(
        const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const override;

    const Teko::LinearOp getInvS(
        const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const override;

    void initializeFromParameterList(
        const Teuchos::ParameterList& lulist, const Teko::InverseLibrary& invLib) override;

   private:
    void initialize_state(
        const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const;

    Teuchos::RCP<Teko::InverseFactory> inv_factory_f_;
    Teuchos::RCP<Teko::InverseFactory> inv_factory_s_;

    // Sparse approximate inverse parameters
    double drop_tol_;
    int fill_level_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
