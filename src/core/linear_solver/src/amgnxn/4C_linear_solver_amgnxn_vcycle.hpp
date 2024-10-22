// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_AMGNXN_VCYCLE_HPP
#define FOUR_C_LINEAR_SOLVER_AMGNXN_VCYCLE_HPP

// Trilinos includes
#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_amgnxn_smoothers.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>
#include <MueLu.hpp>
#include <MueLu_BaseClass.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <MueLu_Utilities.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


namespace Core::LinearSolver::AMGNxN
{
  class Vcycle : public GenericSmoother
  {
   public:
    Vcycle(int NumLevels, int NumSweeps, int FirstLevel);

    void set_operators(std::vector<Teuchos::RCP<BlockedMatrix>> Avec);
    void set_projectors(std::vector<Teuchos::RCP<BlockedMatrix>> Pvec);
    void set_restrictors(std::vector<Teuchos::RCP<BlockedMatrix>> Rvec);
    void set_pre_smoothers(std::vector<Teuchos::RCP<GenericSmoother>> SvecPre);
    void set_pos_smoothers(std::vector<Teuchos::RCP<GenericSmoother>> SvecPos);

    void solve(
        const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero = false) const override;

   private:
    void do_vcycle(
        const BlockedVector& X, BlockedVector& Y, int level, bool InitialGuessIsZero) const;

    int num_levels_;
    int num_sweeps_;
    int first_level_;

    std::vector<Teuchos::RCP<BlockedMatrix>> avec_;
    std::vector<Teuchos::RCP<BlockedMatrix>> pvec_;
    std::vector<Teuchos::RCP<BlockedMatrix>> rvec_;
    std::vector<Teuchos::RCP<GenericSmoother>> svec_pre_;
    std::vector<Teuchos::RCP<GenericSmoother>> svec_pos_;

    bool flag_set_up_a_;
    bool flag_set_up_p_;
    bool flag_set_up_r_;
    bool flag_set_up_pre_;
    bool flag_set_up_pos_;
  };

  // This could be done better with templates
  class VcycleSingle : public SingleFieldSmoother
  {
   public:
    VcycleSingle(int NumLevels, int NumSweeps, int FirstLevel);

    void set_operators(std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Avec);
    void set_projectors(std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Pvec);
    void set_restrictors(std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Rvec);
    void set_pre_smoothers(std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPre);
    void set_pos_smoothers(std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPos);

    void apply(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        bool InitialGuessIsZero) const override;

   private:
    void do_vcycle(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        int level, bool InitialGuessIsZero) const;

    int num_levels_;
    int num_sweeps_;
    int first_level_;

    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> avec_;
    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> pvec_;
    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> rvec_;
    std::vector<Teuchos::RCP<SingleFieldSmoother>> svec_pre_;
    std::vector<Teuchos::RCP<SingleFieldSmoother>> svec_pos_;

    bool flag_set_up_a_;
    bool flag_set_up_p_;
    bool flag_set_up_r_;
    bool flag_set_up_pre_;
    bool flag_set_up_pos_;
  };
}  // namespace Core::LinearSolver::AMGNxN

FOUR_C_NAMESPACE_CLOSE

#endif
