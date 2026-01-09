// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_MATRIXFREE_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian_base.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>
#include <NOX_Thyra_Group.H>
#include <NOX_Thyra_MatrixFreeJacobianOperator.hpp>
#include <Thyra_StateFuncModelEvaluatorBase.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class MatrixFree : public NOX::Nln::Interface::JacobianBase
    {
      class ThyraModelWrapper : public ::Thyra::StateFuncModelEvaluatorBase<double>
      {
       public:
        ThyraModelWrapper(const std::shared_ptr<NOX::Nln::Interface::RequiredBase> model,
            const std::shared_ptr<const Core::LinAlg::Map> map);

        ::Thyra::ModelEvaluatorBase::InArgs<double> getNominalValues() const override;

        ::Thyra::ModelEvaluatorBase::InArgs<double> createInArgs() const override;

        Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> get_x_space() const override;

        Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> get_f_space() const override;

        Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> get_p_space(int l) const override;

        Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> get_g_space(int j) const override;

        ::Thyra::ModelEvaluatorBase::OutArgs<double> createOutArgsImpl() const override;

        void evalModelImpl(const ::Thyra::ModelEvaluatorBase::InArgs<double>& inArgs,
            const ::Thyra::ModelEvaluatorBase::OutArgs<double>& outArgs) const override;

       private:
        const std::shared_ptr<NOX::Nln::Interface::RequiredBase> model_;
        const std::shared_ptr<const Core::LinAlg::Map> map_;

        ::Thyra::ModelEvaluatorBase::InArgs<double> prototype_in_args_;
        ::Thyra::ModelEvaluatorBase::OutArgs<double> prototype_out_args_;

        Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> vector_space_;
      };

      class SparseOperatorWrapper : public Core::LinAlg::SparseOperator
      {
       public:
        SparseOperatorWrapper(const ::Thyra::LinearOpBase<double>& op,
            const std::shared_ptr<const Core::LinAlg::Map> map);

        // Methods of Core::LinAlg::SparseOperator interface
        Epetra_Operator& epetra_operator() override;

        void zero() override;

        void reset() override;

        MPI_Comm get_comm() const override;

        void assemble(int eid, const std::vector<int>& lmstride,
            const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
            const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) override;

        void assemble(double val, int rgid, int cgid) override;

        bool filled() const override;

        void complete(Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

        void complete(const Core::LinAlg::Map& domainmap, const Core::LinAlg::Map& rangemap,
            Core::LinAlg::OptionsMatrixComplete options_matrix_complete = {}) override;

        void un_complete() override;

        void apply_dirichlet(
            const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock = true) override;

        void apply_dirichlet(const Core::LinAlg::Map& dbcmap, bool diagonalblock = true) override;


        const Core::LinAlg::Map& domain_map() const override;

        void add(const Core::LinAlg::SparseOperator& A, const bool transposeA, const double scalarA,
            const double scalarB) override;

        void scale(double ScalarConstant) override;

        void multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
            Core::LinAlg::MultiVector<double>& Y) const override;

       private:
        const ::Thyra::LinearOpBase<double>& operator_;
        const std::shared_ptr<const Core::LinAlg::Map> map_;
      };

     public:
      MatrixFree(Teuchos::ParameterList& printParams,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase>& required,
          const NOX::Nln::Vector& cloneVector, double lambda);

      Core::LinAlg::SparseOperator& get_operator();

      bool compute_jacobian(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override;

     private:
      std::shared_ptr<const Core::LinAlg::Map> build_map(const NOX::Nln::Vector& cloneVector);

      ::NOX::Thyra::MatrixFreeJacobianOperator<double> matrix_free_;
      std::shared_ptr<const Core::LinAlg::Map> map_;
      ThyraModelWrapper thyra_model_wrapper_;
      SparseOperatorWrapper sparse_operator_wrapper_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
